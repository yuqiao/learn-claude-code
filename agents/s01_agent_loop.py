#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while tool_calls:
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.

Usage:
    python s01_agent_loop.py        # normal mode
    python s01_agent_loop.py -v     # verbose mode (print API calls)
"""

import argparse
import os
import subprocess
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# 添加 agents 目录到 path，支持从项目根目录运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verbose_callback import VerboseCallbackHandler

# 解析命令行参数
parser = argparse.ArgumentParser(description="Agent Loop with optional verbose logging")
parser.add_argument("-v", "--verbose", action="store_true", help="Print API request/response")
args = parser.parse_args()
VERBOSE = args.verbose

load_dotenv(override=True)

MODEL = os.environ["MODEL_ID"]

# LangChain automatically reads OPENAI_API_KEY and OPENAI_API_BASE from env
llm = ChatOpenAI(model=MODEL)

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# ANSI 颜色码
COLOR = {
    "reset": "\033[0m",
    "gray": "\033[90m",
    "yellow": "\033[33m",
}


@tool
def bash(command: str) -> str:
    """Run a shell command."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list, verbose_handler: VerboseCallbackHandler | None = None):
    """Agent 循环：执行工具直到模型停止调用工具"""
    llm_with_tools = llm.bind_tools([bash])
    config = {"callbacks": [verbose_handler]} if verbose_handler else None

    while True:
        response = llm_with_tools.invoke(messages, config=config)
        messages.append(response)

        # 如果模型没有调用工具，结束循环
        if not response.tool_calls:
            return

        # 执行每个工具调用，收集结果
        for tool_call in response.tool_calls:
            # 执行命令
            print(f"{COLOR['yellow']}$ {tool_call['args']['command']}{COLOR['reset']}")
            output = bash.invoke(tool_call["args"])

            # 非 verbose 模式下打印截断输出
            if not verbose_handler:
                print(output[:200])

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
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append(HumanMessage(content=query))
        agent_loop(history, verbose_handler)
        # Print the final response
        last_msg = history[-1]
        if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
            print(last_msg.content)
        print()