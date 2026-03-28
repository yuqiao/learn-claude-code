#!/usr/bin/env python3
# Harness: all mechanisms combined -- the complete cockpit for the model.
"""
s_full.py - Full Reference Agent

Capstone implementation combining every mechanism from s01-s11.
Session s12 (task-aware worktree isolation) is taught separately.
NOT a teaching session -- this is the "put it all together" reference.

    +------------------------------------------------------------------+
    |                        FULL AGENT                                 |
    |                                                                   |
    |  System prompt (s05 skills, task-first + optional todo nag)      |
    |                                                                   |
    |  Before each LLM call:                                            |
    |  +--------------------+  +------------------+  +--------------+  |
    |  | Microcompact (s06) |  | Drain bg (s08)   |  | Check inbox  |  |
    |  | Auto-compact (s06) |  | notifications    |  | (s09)        |  |
    |  +--------------------+  +------------------+  +--------------+  |
    |                                                                   |
    |  Tool dispatch (s02 pattern):                                     |
    |  +--------+----------+----------+---------+-----------+          |
    |  | bash   | read     | write    | edit    | TodoWrite |          |
    |  | task   | load_sk  | compress | bg_run  | bg_check  |          |
    |  | t_crt  | t_get    | t_upd    | t_list  | spawn_tm  |          |
    |  | list_tm| send_msg | rd_inbox | bcast   | shutdown  |          |
    |  | plan   | idle     | claim    |         |           |          |
    |  +--------+----------+----------+---------+-----------+          |
    |                                                                   |
    |  Subagent (s04):  spawn -> work -> return summary                 |
    |  Teammate (s09):  spawn -> work -> idle -> auto-claim (s11)      |
    |  Shutdown (s10):  request_id handshake                            |
    |  Plan gate (s10): submit -> approve/reject                        |
    +------------------------------------------------------------------+

    REPL commands: /compact /tasks /team /inbox
"""

import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

load_dotenv(override=True)

# LangChain automatically reads OPENAI_API_KEY and OPENAI_API_BASE from env
MODEL = os.environ["MODEL_ID"]

WORKDIR = Path.cwd()
llm = ChatOpenAI(model=MODEL)

TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
TASKS_DIR = WORKDIR / ".tasks"
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOKEN_THRESHOLD = 100000
POLL_INTERVAL = 5
IDLE_TIMEOUT = 60

VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"}


# === SECTION: base_tools ===
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


@tool
def bash(command: str) -> str:
    """Run a shell command."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


@tool
def read_file(path: str, limit: int = None) -> str:
    """Read file contents."""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to file."""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in file."""
    try:
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# === SECTION: todos (s03) ===
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        validated, ip = [], 0
        for i, item in enumerate(items):
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()
            af = str(item.get("activeForm", "")).strip()
            if not content:
                raise ValueError(f"Item {i}: content required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {i}: invalid status '{status}'")
            if not af:
                raise ValueError(f"Item {i}: activeForm required")
            if status == "in_progress":
                ip += 1
            validated.append({"content": content, "status": status, "activeForm": af})
        if len(validated) > 20:
            raise ValueError("Max 20 todos")
        if ip > 1:
            raise ValueError("Only one in_progress allowed")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            m = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}.get(item["status"], "[?]")
            suffix = f" <- {item['activeForm']}" if item["status"] == "in_progress" else ""
            lines.append(f"{m} {item['content']}{suffix}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)

    def has_open_items(self) -> bool:
        return any(item.get("status") != "completed" for item in self.items)


TODO = TodoManager()


@tool
def TodoWrite(items: list) -> str:
    """Update task tracking list."""
    return TODO.update(items)


# === SECTION: subagent (s04) ===
@tool
def task(prompt: str, agent_type: str = "Explore") -> str:
    """Spawn a subagent for isolated exploration or work."""
    sub_tools = [bash, read_file]
    if agent_type != "Explore":
        sub_tools += [write_file, edit_file]
    sub_llm = ChatOpenAI(model=MODEL)
    sub_llm_with_tools = sub_llm.bind_tools(sub_tools)
    sub_dispatch = {t.name: t for t in sub_tools}
    sub_msgs = [HumanMessage(content=prompt)]
    resp = None
    for _ in range(30):
        resp = sub_llm_with_tools.invoke(sub_msgs)
        sub_msgs.append(resp)
        if not resp.tool_calls:
            break
        for tc in resp.tool_calls:
            h = sub_dispatch.get(tc["name"])
            output = h.invoke(tc["args"]) if h else "Unknown tool"
            sub_msgs.append(ToolMessage(content=str(output)[:50000], tool_call_id=tc["id"]))
    if resp:
        return resp.content or "(no summary)"
    return "(subagent failed)"


# === SECTION: skills (s05) ===
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills = {}
        if skills_dir.exists():
            for f in sorted(skills_dir.rglob("SKILL.md")):
                text = f.read_text()
                match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
                meta, body = {}, text
                if match:
                    for line in match.group(1).strip().splitlines():
                        if ":" in line:
                            k, v = line.split(":", 1)
                            meta[k.strip()] = v.strip()
                    body = match.group(2).strip()
                name = meta.get("name", f.parent.name)
                self.skills[name] = {"meta": meta, "body": body}

    def descriptions(self) -> str:
        if not self.skills:
            return "(no skills)"
        return "\n".join(f"  - {n}: {s['meta'].get('description', '-')}" for n, s in self.skills.items())

    def load(self, name: str) -> str:
        s = self.skills.get(name)
        if not s:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{s['body']}\n</skill>"


SKILLS = SkillLoader(SKILLS_DIR)


@tool
def load_skill(name: str) -> str:
    """Load specialized knowledge by name."""
    return SKILLS.load(name)


# === SECTION: compression (s06) ===
def estimate_tokens(messages: list) -> int:
    return len(json.dumps([m.model_dump() if hasattr(m, 'model_dump') else str(m) for m in messages], default=str)) // 4


def microcompact(messages: list):
    tool_results = [(i, msg) for i, msg in enumerate(messages) if isinstance(msg, ToolMessage)]
    if len(tool_results) <= 3:
        return
    for idx, msg in tool_results[:-3]:
        if len(msg.content) > 100:
            messages[idx] = ToolMessage(content="[cleared]", tool_call_id=msg.tool_call_id)


def auto_compact(messages: list) -> list:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg.model_dump() if hasattr(msg, 'model_dump') else str(msg), default=str) + "\n")
    conv_text = json.dumps([m.model_dump() if hasattr(m, 'model_dump') else str(m) for m in messages], default=str)[:80000]
    summary_llm = ChatOpenAI(model=MODEL)
    resp = summary_llm.invoke([HumanMessage(content=f"Summarize for continuity:\n{conv_text}")])
    return [
        HumanMessage(content=f"[Compressed. Transcript: {path}]\n{resp.content}"),
        AIMessage(content="Understood. Continuing with summary context."),
    ]


@tool
def compress() -> str:
    """Manually compress conversation context."""
    return "Compressing..."


# === SECTION: file_tasks (s07) ===
class TaskManager:
    def __init__(self):
        TASKS_DIR.mkdir(exist_ok=True)

    def _next_id(self) -> int:
        ids = [int(f.stem.split("_")[1]) for f in TASKS_DIR.glob("task_*.json")]
        return max(ids, default=0) + 1

    def _load(self, tid: int) -> dict:
        p = TASKS_DIR / f"task_{tid}.json"
        if not p.exists():
            raise ValueError(f"Task {tid} not found")
        return json.loads(p.read_text())

    def _save(self, task: dict):
        (TASKS_DIR / f"task_{task['id']}.json").write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        task = {"id": self._next_id(), "subject": subject, "description": description,
                "status": "pending", "owner": None, "blockedBy": [], "blocks": []}
        self._save(task)
        return json.dumps(task, indent=2)

    def get(self, tid: int) -> str:
        return json.dumps(self._load(tid), indent=2)

    def update(self, tid: int, status: str = None, add_blocked_by: list = None, add_blocks: list = None) -> str:
        task = self._load(tid)
        if status:
            task["status"] = status
            if status == "completed":
                for f in TASKS_DIR.glob("task_*.json"):
                    t = json.loads(f.read_text())
                    if tid in t.get("blockedBy", []):
                        t["blockedBy"].remove(tid)
                        self._save(t)
            if status == "deleted":
                (TASKS_DIR / f"task_{tid}.json").unlink(missing_ok=True)
                return f"Task {tid} deleted"
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))
        self._save(task)
        return json.dumps(task, indent=2)

    def list_all(self) -> str:
        tasks = [json.loads(f.read_text()) for f in sorted(TASKS_DIR.glob("task_*.json"))]
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            m = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            owner = f" @{t['owner']}" if t.get("owner") else ""
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{m} #{t['id']}: {t['subject']}{owner}{blocked}")
        return "\n".join(lines)

    def claim(self, tid: int, owner: str) -> str:
        task = self._load(tid)
        task["owner"] = owner
        task["status"] = "in_progress"
        self._save(task)
        return f"Claimed task #{tid} for {owner}"


TASK_MGR = TaskManager()


@tool
def task_create(subject: str, description: str = "") -> str:
    """Create a persistent file task."""
    return TASK_MGR.create(subject, description)


@tool
def task_get(task_id: int) -> str:
    """Get task details by ID."""
    return TASK_MGR.get(task_id)


@tool
def task_update(task_id: int, status: str = None, add_blocked_by: list = None, add_blocks: list = None) -> str:
    """Update task status or dependencies."""
    return TASK_MGR.update(task_id, status, add_blocked_by, add_blocks)


@tool
def task_list() -> str:
    """List all tasks."""
    return TASK_MGR.list_all()


@tool
def claim_task(task_id: int) -> str:
    """Claim a task from the board."""
    return TASK_MGR.claim(task_id, "lead")


# === SECTION: background (s08) ===
class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self.notifications = []

    def run(self, command: str, timeout: int = 120) -> str:
        tid = str(uuid.uuid4())[:8]
        self.tasks[tid] = {"status": "running", "command": command, "result": None}
        threading.Thread(target=self._exec, args=(tid, command, timeout), daemon=True).start()
        return f"Background task {tid} started: {command[:80]}"

    def _exec(self, tid: str, command: str, timeout: int):
        try:
            r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=timeout)
            output = (r.stdout + r.stderr).strip()[:50000]
            self.tasks[tid].update({"status": "completed", "result": output or "(no output)"})
        except Exception as e:
            self.tasks[tid].update({"status": "error", "result": str(e)})
        self.notifications.append({"task_id": tid, "status": self.tasks[tid]["status"], "result": self.tasks[tid]["result"][:500]})

    def check(self, tid: str = None) -> str:
        if tid:
            t = self.tasks.get(tid)
            return f"[{t['status']}] {t.get('result', '(running)')}" if t else f"Unknown: {tid}"
        return "\n".join(f"{k}: [{v['status']}] {v['command'][:60]}" for k, v in self.tasks.items()) or "No bg tasks."

    def drain(self) -> list:
        notifs = list(self.notifications)
        self.notifications.clear()
        return notifs


BG = BackgroundManager()


@tool
def background_run(command: str, timeout: int = 120) -> str:
    """Run command in background thread."""
    return BG.run(command, timeout)


@tool
def check_background(task_id: str = None) -> str:
    """Check background task status."""
    return BG.check(task_id)


# === SECTION: messaging (s09) ===
class MessageBus:
    def __init__(self):
        INBOX_DIR.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str, msg_type: str = "message", extra: dict = None) -> str:
        msg = {"type": msg_type, "from": sender, "content": content, "timestamp": time.time()}
        if extra:
            msg.update(extra)
        with open(INBOX_DIR / f"{to}.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        path = INBOX_DIR / f"{name}.jsonl"
        if not path.exists():
            return []
        msgs = [json.loads(l) for l in path.read_text().strip().splitlines() if l]
        path.write_text("")
        return msgs

    def broadcast(self, sender: str, content: str, names: list) -> str:
        count = sum(1 for n in names if n != sender and self.send(sender, n, content, "broadcast"))
        return f"Broadcast to {count} teammates"


BUS = MessageBus()


# === SECTION: shutdown + plan tracking (s10) ===
shutdown_requests = {}
plan_requests = {}


def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send("lead", teammate, "Please shut down.", "shutdown_request", {"request_id": req_id})
    return f"Shutdown request {req_id} sent to '{teammate}'"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    req["status"] = "approved" if approve else "rejected"
    BUS.send("lead", req["from"], feedback, "plan_approval_response",
             {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"Plan {req['status']} for '{req['from']}'"


@tool
def shutdown_request(teammate: str) -> str:
    """Request a teammate to shut down."""
    return handle_shutdown_request(teammate)


@tool
def plan_approval(request_id: str, approve: bool, feedback: str = "") -> str:
    """Approve or reject a teammate's plan."""
    return handle_plan_review(request_id, approve, feedback)


# === SECTION: team (s09/s11) ===
class TeammateManager:
    def __init__(self, bus: MessageBus, task_mgr: TaskManager):
        TEAM_DIR.mkdir(exist_ok=True)
        self.bus = bus
        self.task_mgr = task_mgr
        self.config_path = TEAM_DIR / "config.json"
        self.config = self._load()
        self.threads = {}

    def _load(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save()
        threading.Thread(target=self._loop, args=(name, role, prompt), daemon=True).start()
        return f"Spawned '{name}' (role: {role})"

    def _set_status(self, name: str, status: str):
        member = self._find(name)
        if member:
            member["status"] = status
            self._save()

    def _loop(self, name: str, role: str, prompt: str):
        team_name = self.config["team_name"]
        sys_prompt = (f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
                      f"Use idle when done with current work. You may auto-claim tasks.")
        messages = [SystemMessage(content=sys_prompt), HumanMessage(content=prompt)]
        tools = [bash, read_file, write_file, edit_file,
                 type("Tool", (), {"name": "send_message", "invoke": lambda s, a: self.bus.send(name, a["to"], a["content"])})(),
                 type("Tool", (), {"name": "idle", "invoke": lambda s, a: "Entering idle phase."})(),
                 type("Tool", (), {"name": "claim_task", "invoke": lambda s, a: self.task_mgr.claim(a["task_id"], name)})()]
        teammate_llm = ChatOpenAI(model=MODEL)
        teammate_llm_with_tools = teammate_llm.bind_tools(tools)

        while True:
            for _ in range(50):
                inbox = self.bus.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
                    messages.append(HumanMessage(content=json.dumps(msg)))
                try:
                    response = teammate_llm_with_tools.invoke(messages)
                except Exception:
                    self._set_status(name, "shutdown")
                    return
                messages.append(response)
                if not response.tool_calls:
                    break
                idle_requested = False
                for tc in response.tool_calls:
                    if tc["name"] == "idle":
                        idle_requested = True
                        output = "Entering idle phase."
                    elif tc["name"] == "claim_task":
                        output = self.task_mgr.claim(tc["args"]["task_id"], name)
                    elif tc["name"] == "send_message":
                        output = self.bus.send(name, tc["args"]["to"], tc["args"]["content"])
                    else:
                        dispatch = {"bash": bash, "read_file": read_file, "write_file": write_file, "edit_file": edit_file}
                        h = dispatch.get(tc["name"])
                        output = h.invoke(tc["args"]) if h else "Unknown"
                    print(f"  [{name}] {tc['name']}: {str(output)[:120]}")
                    messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
                if idle_requested:
                    break

            self._set_status(name, "idle")
            resume = False
            for _ in range(IDLE_TIMEOUT // max(POLL_INTERVAL, 1)):
                time.sleep(POLL_INTERVAL)
                inbox = self.bus.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append(HumanMessage(content=json.dumps(msg)))
                    resume = True
                    break
                unclaimed = [json.loads(f.read_text()) for f in sorted(TASKS_DIR.glob("task_*.json"))
                             if json.loads(f.read_text()).get("status") == "pending"
                             and not json.loads(f.read_text()).get("owner")
                             and not json.loads(f.read_text()).get("blockedBy")]
                if unclaimed:
                    t = unclaimed[0]
                    self.task_mgr.claim(t["id"], name)
                    if len(messages) <= 3:
                        messages.insert(0, HumanMessage(content=f"<identity>You are '{name}', role: {role}, team: {team_name}.</identity>"))
                        messages.insert(1, AIMessage(content=f"I am {name}. Continuing."))
                    messages.append(HumanMessage(content=f"<auto-claimed>Task #{t['id']}: {t['subject']}\n{t.get('description', '')}</auto-claimed>"))
                    messages.append(AIMessage(content=f"Claimed task #{t['id']}. Working on it."))
                    resume = True
                    break
            if not resume:
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(BUS, TASK_MGR)


# === SECTION: system_prompt ===
SYSTEM = f"""You are a coding agent at {WORKDIR}. Use tools to solve tasks.
Prefer task_create/task_update/task_list for multi-step work. Use TodoWrite for short checklists.
Use task for subagent delegation. Use load_skill for specialized knowledge.
Skills: {SKILLS.descriptions()}"""


@tool
def spawn_teammate(name: str, role: str, prompt: str) -> str:
    """Spawn a persistent autonomous teammate."""
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
    """Send message to all teammates."""
    return BUS.broadcast("lead", content, TEAM.member_names())


@tool
def idle() -> str:
    """Enter idle state."""
    return "Lead does not idle."


# === SECTION: tool_dispatch ===
TOOLS = [bash, read_file, write_file, edit_file, TodoWrite, task, load_skill, compress,
         background_run, check_background, task_create, task_get, task_update, task_list,
         spawn_teammate, list_teammates, send_message, read_inbox, broadcast,
         shutdown_request, plan_approval, idle, claim_task]
TOOL_DISPATCH = {t.name: t for t in TOOLS}


# === SECTION: agent_loop ===
def agent_loop(messages: list):
    rounds_without_todo = 0
    while True:
        microcompact(messages)
        if estimate_tokens(messages) > TOKEN_THRESHOLD:
            print("[auto-compact triggered]")
            messages[:] = auto_compact(messages)
        notifs = BG.drain()
        if notifs:
            txt = "\n".join(f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs)
            messages.append(HumanMessage(content=f"<background-results>\n{txt}\n</background-results>"))
            messages.append(AIMessage(content="Noted background results."))
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append(HumanMessage(content=f"<inbox>{json.dumps(inbox, indent=2)}</inbox>"))
            messages.append(AIMessage(content="Noted inbox messages."))
        response = llm.bind_tools(TOOLS).invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return
        used_todo = False
        manual_compress = False
        for tc in response.tool_calls:
            if tc["name"] == "compress":
                manual_compress = True
            handler = TOOL_DISPATCH.get(tc["name"])
            try:
                output = handler.invoke(tc["args"]) if handler else f"Unknown tool: {tc['name']}"
            except Exception as e:
                output = f"Error: {e}"
            print(f"> {tc['name']}: {str(output)[:200]}")
            messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
            if tc["name"] == "TodoWrite":
                used_todo = True
        rounds_without_todo = 0 if used_todo else rounds_without_todo + 1
        if TODO.has_open_items() and rounds_without_todo >= 3:
            messages.append(HumanMessage(content="<reminder>Update your todos.</reminder>"))
        if manual_compress:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


if __name__ == "__main__":
    history = [SystemMessage(content=SYSTEM)]
    while True:
        try:
            query = input("\033[36ms_full >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/compact":
            if history:
                print("[manual compact via /compact]")
                history[:] = auto_compact(history)
            continue
        if query.strip() == "/tasks":
            print(TASK_MGR.list_all())
            continue
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue
        history.append(HumanMessage(content=query))
        agent_loop(history)
        print()