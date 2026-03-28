#!/usr/bin/env python3
# Harness: tool dispatch -- expanding what the model can reach.
"""
s02_tool_use.py - Tools

The agent loop from s01 didn't change. We just added tools to the array
and a dispatch map to route calls.

    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash: run_bash |
                          ^          |   read: run_read |
                          |          |   write: run_wr  |
                          +----------+   edit: run_edit |
                          tool_result| }                |
                                     +------------------+

Key insight: "The loop didn't change at all. I just added tools."

Usage:
    python s02_tool_use.py        # normal mode
    python s02_tool_use.py -v     # verbose mode (print API calls)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# 添加 agents 目录到 path，支持从项目根目录运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verbose_callback import VerboseCallbackHandler

# 解析命令行参数
parser = argparse.ArgumentParser(description="Tool Dispatch with optional verbose logging")
parser.add_argument("-v", "--verbose", action="store_true", help="Print API request/response")
args = parser.parse_args()
VERBOSE = args.verbose

load_dotenv(override=True)

# LangChain automatically reads OPENAI_API_KEY and OPENAI_API_BASE from env
MODEL = os.environ["MODEL_ID"]

WORKDIR = Path.cwd()
llm = ChatOpenAI(model=MODEL)

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


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
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
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
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- Tool list for binding --
TOOLS = [bash, read_file, write_file, edit_file]

# -- Tool dispatch map: {tool_name: handler} --
TOOL_DISPATCH = {t.name: t for t in TOOLS}


def agent_loop(messages: list, verbose_handler: VerboseCallbackHandler | None = None):
    """Agent 循环：执行工具直到模型停止调用工具"""
    llm_with_tools = llm.bind_tools(TOOLS)
    config = {"callbacks": [verbose_handler]} if verbose_handler else None

    while True:
        response = llm_with_tools.invoke(messages, config=config)
        messages.append(response)
        if not response.tool_calls:
            return
        for tool_call in response.tool_calls:
            handler = TOOL_DISPATCH.get(tool_call["name"])
            output = handler.invoke(tool_call["args"]) if handler else f"Unknown tool: {tool_call['name']}"
            # 非 verbose 模式下打印截断输出
            if not verbose_handler:
                print(f"> {tool_call['name']}: {output[:200]}")
            messages.append(ToolMessage(
                content=output,
                tool_call_id=tool_call["id"],
            ))


if __name__ == "__main__":
    # 创建 verbose handler（如果启用）
    verbose_handler = VerboseCallbackHandler() if VERBOSE else None

    history = [SystemMessage(content=SYSTEM)]
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append(HumanMessage(content=query))
        agent_loop(history, verbose_handler)
        last_msg = history[-1]
        if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
            print(last_msg.content)
        print()