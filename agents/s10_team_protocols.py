#!/usr/bin/env python3
# Harness: protocols -- structured handshakes between models.
"""
s10_team_protocols.py - Team Protocols

Shutdown protocol and plan approval protocol, both using the same
request_id correlation pattern. Builds on s09's team messaging.

    Shutdown FSM: pending -> approved | rejected

    Lead                              Teammate
    +---------------------+          +---------------------+
    | shutdown_request     |          |                     |
    | {                    | -------> | receives request    |
    |   request_id: abc    |          | decides: approve?   |
    | }                    |          |                     |
    +---------------------+          +---------------------+
                                             |
    +---------------------+          +-------v-------------+
    | shutdown_response    | <------- | shutdown_response   |
    | {                    |          | {                   |
    |   request_id: abc    |          |   request_id: abc   |
    |   approve: true      |          |   approve: true     |
    | }                    |          | }                   |
    +---------------------+          +---------------------+
            |
            v
    status -> "shutdown", thread stops

    Plan approval FSM: pending -> approved | rejected

    Teammate                          Lead
    +---------------------+          +---------------------+
    | plan_approval        |          |                     |
    | submit: {plan:"..."}| -------> | reviews plan text   |
    +---------------------+          | approve/reject?     |
                                     +---------------------+
                                             |
    +---------------------+          +-------v-------------+
    | plan_approval_resp   | <------- | plan_approval       |
    | {approve: true}      |          | review: {req_id,    |
    +---------------------+          |   approve: true}     |
                                     +---------------------+

    Trackers: {request_id: {"target|from": name, "status": "pending|..."}}

Key insight: "Same request_id correlation pattern, two domains."
"""

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

load_dotenv(override=True)

# LangChain automatically reads OPENAI_API_KEY and OPENAI_API_BASE from env
MODEL = os.environ["MODEL_ID"]

WORKDIR = Path.cwd()
llm = ChatOpenAI(model=MODEL)
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

SYSTEM = f"You are a team lead at {WORKDIR}. Manage teammates with shutdown and plan approval protocols."

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock()


# -- MessageBus: JSONL inbox per teammate --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {"type": msg_type, "from": sender, "content": content, "timestamp": time.time()}
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = [json.loads(line) for line in inbox_path.read_text().strip().splitlines() if line]
        inbox_path.write_text("")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = sum(1 for name in teammates if name != sender and self.send(sender, name, content, "broadcast"))
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


# -- Base tool implementations --
def _safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int = None) -> str:
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- TeammateManager with shutdown + plan approval --
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()
        thread = threading.Thread(target=self._teammate_loop, args=(name, role, prompt), daemon=True)
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        teammate_llm = ChatOpenAI(model=MODEL)
        sys_prompt = (f"You are '{name}', role: {role}, at {WORKDIR}. "
                      f"Submit plans via plan_approval before major work. "
                      f"Respond to shutdown_request with shutdown_response.")
        tools = self._build_teammate_tools()
        tool_dispatch = {t.name: t for t in tools}
        teammate_llm_with_tools = teammate_llm.bind_tools(tools)
        messages = [SystemMessage(content=sys_prompt), HumanMessage(content=prompt)]
        should_exit = False
        for _ in range(50):
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append(HumanMessage(content=json.dumps(msg)))
            if should_exit:
                break
            try:
                response = teammate_llm_with_tools.invoke(messages)
            except Exception:
                break
            messages.append(response)
            if not response.tool_calls:
                break
            for tool_call in response.tool_calls:
                output = self._exec_tool(name, tool_call["name"], tool_call["args"], tool_dispatch)
                print(f"  [{name}] {tool_call['name']}: {str(output)[:120]}")
                messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
                if tool_call["name"] == "shutdown_response" and tool_call["args"].get("approve"):
                    should_exit = True
        member = self._find_member(name)
        if member:
            member["status"] = "shutdown" if should_exit else "idle"
            self._save_config()

    def _exec_tool(self, sender: str, tool_name: str, args: dict, dispatch: dict) -> str:
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            approve = args["approve"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if approve else "rejected"
            BUS.send(sender, "lead", args.get("reason", ""), "shutdown_response",
                     {"request_id": req_id, "approve": approve})
            return f"Shutdown {'approved' if approve else 'rejected'}"
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            BUS.send(sender, "lead", plan_text, "plan_approval_response",
                     {"request_id": req_id, "plan": plan_text})
            return f"Plan submitted (request_id={req_id}). Waiting for lead approval."
        handler = dispatch.get(tool_name)
        if handler:
            return handler.invoke(args)
        return f"Unknown tool: {tool_name}"

    def _build_teammate_tools(self):
        @tool
        def bash(command: str) -> str:
            """Run a shell command."""
            return _run_bash(command)

        @tool
        def read_file(path: str) -> str:
            """Read file contents."""
            return _run_read(path)

        @tool
        def write_file(path: str, content: str) -> str:
            """Write content to file."""
            return _run_write(path, content)

        @tool
        def edit_file(path: str, old_text: str, new_text: str) -> str:
            """Replace exact text in file."""
            return _run_edit(path, old_text, new_text)

        @tool
        def send_message(to: str, content: str, msg_type: str = "message") -> str:
            """Send message to a teammate."""
            return "dispatch"

        @tool
        def read_inbox() -> str:
            """Read and drain your inbox."""
            return "dispatch"

        @tool
        def shutdown_response(request_id: str, approve: bool, reason: str = "") -> str:
            """Respond to a shutdown request. Approve to shut down, reject to keep working."""
            return "dispatch"

        @tool
        def plan_approval(plan: str) -> str:
            """Submit a plan for lead approval. Provide plan text."""
            return "dispatch"

        return [bash, read_file, write_file, edit_file, send_message, read_inbox, shutdown_response, plan_approval]

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(TEAM_DIR)


# -- Lead tools --
@tool
def bash(command: str) -> str:
    """Run a shell command."""
    return _run_bash(command)


@tool
def read_file(path: str, limit: int = None) -> str:
    """Read file contents."""
    return _run_read(path, limit)


@tool
def write_file(path: str, content: str) -> str:
    """Write content to file."""
    return _run_write(path, content)


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in file."""
    return _run_edit(path, old_text, new_text)


@tool
def spawn_teammate(name: str, role: str, prompt: str) -> str:
    """Spawn a persistent teammate."""
    return TEAM.spawn(name, role, prompt)


@tool
def list_teammates() -> str:
    """List all teammates."""
    return TEAM.list_all()


@tool
def send_message(to: str, content: str, msg_type: str = "message") -> str:
    """Send a message to a teammate."""
    return BUS.send("lead", to, content, msg_type)


@tool
def read_inbox() -> str:
    """Read and drain the lead's inbox."""
    return json.dumps(BUS.read_inbox("lead"), indent=2)


@tool
def broadcast(content: str) -> str:
    """Send a message to all teammates."""
    return BUS.broadcast("lead", content, TEAM.member_names())


@tool
def shutdown_request(teammate: str) -> str:
    """Request a teammate to shut down gracefully. Returns a request_id for tracking."""
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send("lead", teammate, "Please shut down gracefully.", "shutdown_request", {"request_id": req_id})
    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)"


@tool
def shutdown_response(request_id: str) -> str:
    """Check the status of a shutdown request by request_id."""
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))


@tool
def plan_approval(request_id: str, approve: bool, feedback: str = "") -> str:
    """Approve or reject a teammate's plan. Provide request_id + approve + optional feedback."""
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    BUS.send("lead", req["from"], feedback, "plan_approval_response",
             {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"Plan {req['status']} for '{req['from']}'"


TOOLS = [bash, read_file, write_file, edit_file, spawn_teammate, list_teammates,
         send_message, read_inbox, broadcast, shutdown_request, shutdown_response, plan_approval]
TOOL_DISPATCH = {t.name: t for t in TOOLS}


def agent_loop(messages: list):
    llm_with_tools = llm.bind_tools(TOOLS)
    while True:
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append(HumanMessage(content=f"<inbox>{json.dumps(inbox, indent=2)}</inbox>"))
            messages.append(AIMessage(content="Noted inbox messages."))
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return
        for tool_call in response.tool_calls:
            handler = TOOL_DISPATCH.get(tool_call["name"])
            try:
                output = handler.invoke(tool_call["args"]) if handler else f"Unknown tool: {tool_call['name']}"
            except Exception as e:
                output = f"Error: {e}"
            print(f"> {tool_call['name']}: {str(output)[:200]}")
            messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))


if __name__ == "__main__":
    history = [SystemMessage(content=SYSTEM)]
    while True:
        try:
            query = input("\033[36ms10 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue
        history.append(HumanMessage(content=query))
        agent_loop(history)
        last_msg = history[-1]
        if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
            print(last_msg.content)
        print()