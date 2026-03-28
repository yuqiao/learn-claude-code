#!/usr/bin/env python3
# Harness: autonomy -- models that find work without being told.
"""
s11_autonomous_agents.py - Autonomous Agents

Idle cycle with task board polling, auto-claiming unclaimed tasks, and
identity re-injection after context compression. Builds on s10's protocols.

    Teammate lifecycle:
    +-------+
    | spawn |
    +---+---+
        |
        v
    +-------+  tool_use    +-------+
    | WORK  | <----------- |  LLM  |
    +---+---+              +-------+
        |
        | stop_reason != tool_use
        v
    +--------+
    | IDLE   | poll every 5s for up to 60s
    +---+----+
        |
        +---> check inbox -> message? -> resume WORK
        |
        +---> scan .tasks/ -> unclaimed? -> claim -> resume WORK
        |
        +---> timeout (60s) -> shutdown

    Identity re-injection after compression:
    messages = [identity_block, ...remaining...]
    "You are 'coder', role: backend, team: my-team"

Key insight: "The agent finds work itself."
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
TASKS_DIR = WORKDIR / ".tasks"

POLL_INTERVAL = 5
IDLE_TIMEOUT = 60

SYSTEM = f"You are a team lead at {WORKDIR}. Teammates are autonomous -- they find work themselves."

VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"}

shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock()
_claim_lock = threading.Lock()


# -- MessageBus: JSONL inbox per teammate --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str, msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {"type": msg_type, "from": sender, "content": content, "timestamp": time.time()}
        if extra:
            msg.update(extra)
        with open(self.dir / f"{to}.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        path = self.dir / f"{name}.jsonl"
        if not path.exists():
            return []
        msgs = [json.loads(line) for line in path.read_text().strip().splitlines() if line]
        path.write_text("")
        return msgs

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = sum(1 for n in teammates if n != sender and self.send(sender, n, content, "broadcast"))
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


# -- Task board scanning --
def scan_unclaimed_tasks() -> list:
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if task.get("status") == "pending" and not task.get("owner") and not task.get("blockedBy"):
            unclaimed.append(task)
    return unclaimed


def claim_task(task_id: int, owner: str) -> str:
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text())
        task["owner"] = owner
        task["status"] = "in_progress"
        path.write_text(json.dumps(task, indent=2))
    return f"Claimed task #{task_id} for {owner}"


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


def _run_read(path: str) -> str:
    try:
        return _safe_path(path).read_text()[:50000]
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


# -- Autonomous TeammateManager --
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

    def _set_status(self, name: str, status: str):
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()

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
        thread = threading.Thread(target=self._loop, args=(name, role, prompt), daemon=True)
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _loop(self, name: str, role: str, prompt: str):
        team_name = self.config["team_name"]
        sys_prompt = (f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
                      f"Use idle tool when you have no more work. You will auto-claim new tasks.")
        messages = [SystemMessage(content=sys_prompt), HumanMessage(content=prompt)]
        tools = self._build_teammate_tools()
        tool_dispatch = {t.name: t for t in tools}
        teammate_llm = ChatOpenAI(model=MODEL)
        teammate_llm_with_tools = teammate_llm.bind_tools(tools)

        while True:
            # -- WORK PHASE --
            for _ in range(50):
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
                    messages.append(HumanMessage(content=json.dumps(msg)))
                try:
                    response = teammate_llm_with_tools.invoke(messages)
                except Exception:
                    self._set_status(name, "idle")
                    return
                messages.append(response)
                if not response.tool_calls:
                    break
                idle_requested = False
                for tool_call in response.tool_calls:
                    if tool_call["name"] == "idle":
                        idle_requested = True
                        output = "Entering idle phase. Will poll for new tasks."
                    else:
                        output = self._exec_tool(name, tool_call["name"], tool_call["args"], tool_dispatch)
                    print(f"  [{name}] {tool_call['name']}: {str(output)[:120]}")
                    messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
                if idle_requested:
                    break

            # -- IDLE PHASE --
            self._set_status(name, "idle")
            resume = False
            for _ in range(IDLE_TIMEOUT // max(POLL_INTERVAL, 1)):
                time.sleep(POLL_INTERVAL)
                inbox = BUS.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append(HumanMessage(content=json.dumps(msg)))
                    resume = True
                    break
                unclaimed = scan_unclaimed_tasks()
                if unclaimed:
                    task = unclaimed[0]
                    claim_task(task["id"], name)
                    if len(messages) <= 3:
                        messages.insert(0, HumanMessage(content=f"<identity>You are '{name}', role: {role}, team: {team_name}.</identity>"))
                        messages.insert(1, AIMessage(content=f"I am {name}. Continuing."))
                    messages.append(HumanMessage(content=f"<auto-claimed>Task #{task['id']}: {task['subject']}\n{task.get('description', '')}</auto-claimed>"))
                    messages.append(AIMessage(content=f"Claimed task #{task['id']}. Working on it."))
                    resume = True
                    break

            if not resume:
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")

    def _exec_tool(self, sender: str, tool_name: str, args: dict, dispatch: dict) -> str:
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if args["approve"] else "rejected"
            BUS.send(sender, "lead", args.get("reason", ""), "shutdown_response",
                     {"request_id": req_id, "approve": args["approve"]})
            return f"Shutdown {'approved' if args['approve'] else 'rejected'}"
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            BUS.send(sender, "lead", plan_text, "plan_approval_response",
                     {"request_id": req_id, "plan": plan_text})
            return f"Plan submitted (request_id={req_id}). Waiting for approval."
        if tool_name == "claim_task":
            return claim_task(args["task_id"], sender)
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
            """Respond to a shutdown request."""
            return "dispatch"

        @tool
        def plan_approval(plan: str) -> str:
            """Submit a plan for lead approval."""
            return "dispatch"

        @tool
        def idle() -> str:
            """Signal that you have no more work. Enters idle polling phase."""
            return "Entering idle phase."

        @tool
        def claim_task(task_id: int) -> str:
            """Claim a task from the task board by ID."""
            return "dispatch"

        return [bash, read_file, write_file, edit_file, send_message, read_inbox,
                shutdown_response, plan_approval, idle, claim_task]

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
def spawn_teammate(name: str, role: str, prompt: str) -> str:
    """Spawn an autonomous teammate."""
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
    """Request a teammate to shut down."""
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send("lead", teammate, "Please shut down.", "shutdown_request", {"request_id": req_id})
    return f"Shutdown request {req_id} sent to '{teammate}'"


@tool
def shutdown_response(request_id: str) -> str:
    """Check shutdown request status."""
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))


@tool
def plan_approval(request_id: str, approve: bool, feedback: str = "") -> str:
    """Approve or reject a teammate's plan."""
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    BUS.send("lead", req["from"], feedback, "plan_approval_response",
             {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"Plan {req['status']} for '{req['from']}'"


@tool
def idle() -> str:
    """Enter idle state (for lead -- rarely used)."""
    return "Lead does not idle."


@tool
def claim_task(task_id: int) -> str:
    """Claim a task from the board by ID."""
    return claim_task(task_id, "lead")


TOOLS = [bash, read_file, write_file, edit_file, spawn_teammate, list_teammates,
         send_message, read_inbox, broadcast, shutdown_request, shutdown_response,
         plan_approval, idle, claim_task]
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
            query = input("\033[36ms11 >> \033[0m")
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
        if query.strip() == "/tasks":
            TASKS_DIR.mkdir(exist_ok=True)
            for f in sorted(TASKS_DIR.glob("task_*.json")):
                t = json.loads(f.read_text())
                marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
                owner = f" @{t['owner']}" if t.get("owner") else ""
                print(f"  {marker} #{t['id']}: {t['subject']}{owner}")
            continue
        history.append(HumanMessage(content=query))
        agent_loop(history)
        last_msg = history[-1]
        if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
            print(last_msg.content)
        print()