#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions.
"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.

Key insight: "The agent can forget strategically and keep working forever."
"""

import json
import os
import subprocess
import time
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

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4


# -- Layer 1: micro_compact - replace old tool results with placeholders --
def micro_compact(messages: list) -> list:
    # Collect ToolMessage entries
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            tool_results.append((msg_idx, msg))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    # Find tool_name for each result by matching tool_call_id in prior AIMessage
    tool_name_map = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name_map[tc["id"]] = tc["name"]
    # Clear old results (keep last KEEP_RECENT)
    to_clear = tool_results[:-KEEP_RECENT]
    for idx, msg in to_clear:
        if len(msg.content) > 100:
            tool_name = tool_name_map.get(msg.tool_call_id, "unknown")
            messages[idx] = ToolMessage(
                content=f"[Previous: used {tool_name}]",
                tool_call_id=msg.tool_call_id,
            )
    return messages


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list) -> list:
    # Save full transcript to disk
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg.model_dump() if hasattr(msg, 'model_dump') else str(msg)) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    # Ask LLM to summarize
    conversation_text = str(messages)[:80000]
    summary_llm = ChatOpenAI(model=MODEL)
    response = summary_llm.invoke([
        HumanMessage(content=(
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text
        ))
    ])
    summary = response.content
    # Replace all messages with compressed summary
    return [
        HumanMessage(content=f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"),
        AIMessage(content="Understood. I have the context from the summary. Continuing."),
    ]


# -- Tool implementations --
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
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
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
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in file."""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def compact(focus: str = "") -> str:
    """Trigger manual conversation compression."""
    return "Manual compression requested."


TOOLS = [bash, read_file, write_file, edit_file, compact]
TOOL_DISPATCH = {t.name: t for t in TOOLS}


def agent_loop(messages: list):
    llm_with_tools = llm.bind_tools(TOOLS)
    while True:
        # Layer 1: micro_compact before each LLM call
        micro_compact(messages)
        # Layer 2: auto_compact if token estimate exceeds threshold
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return
        manual_compact = False
        for tool_call in response.tool_calls:
            if tool_call["name"] == "compact":
                manual_compact = True
                output = "Compressing..."
            else:
                handler = TOOL_DISPATCH.get(tool_call["name"])
                try:
                    output = handler.invoke(tool_call["args"]) if handler else f"Unknown tool: {tool_call['name']}"
                except Exception as e:
                    output = f"Error: {e}"
            print(f"> {tool_call['name']}: {str(output)[:200]}")
            messages.append(ToolMessage(
                content=str(output),
                tool_call_id=tool_call["id"],
            ))
        # Layer 3: manual compact triggered by the compact tool
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


if __name__ == "__main__":
    history = [SystemMessage(content=SYSTEM)]
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append(HumanMessage(content=query))
        agent_loop(history)
        last_msg = history[-1]
        if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
            print(last_msg.content)
        print()