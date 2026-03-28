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
import json
import os
import subprocess

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

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
    "blue": "\033[34m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
}


def highlight_json(text: str) -> str:
    """简单的 JSON 语法高亮"""
    result = []
    i = 0
    in_string = False
    string_char = None

    while i < len(text):
        char = text[i]

        # 处理字符串
        if char in '"\'':
            if not in_string:
                in_string = True
                string_char = char
                result.append(f"{COLOR['green']}{char}")
            elif char == string_char and (i == 0 or text[i-1] != '\\'):
                in_string = False
                string_char = None
                result.append(f"{char}{COLOR['reset']}")
            else:
                result.append(char)
            i += 1
            continue

        if in_string:
            result.append(char)
            i += 1
            continue

        # 处理关键字
        if char.isdigit() or (char == '-' and i + 1 < len(text) and text[i + 1].isdigit()):
            j = i
            while j < len(text) and (text[j].isdigit() or text[j] in '.-eE+'):
                j += 1
            result.append(f"{COLOR['magenta']}{text[i:j]}{COLOR['reset']}")
            i = j
            continue

        if text[i:i+4] in ('true', 'null'):
            result.append(f"{COLOR['cyan']}{text[i:i+4]}{COLOR['reset']}")
            i += 4
            continue

        if text[i:i+5] == 'false':
            result.append(f"{COLOR['cyan']}{text[i:i+5]}{COLOR['reset']}")
            i += 5
            continue

        # 处理标点
        if char in '{}[]':
            result.append(f"{COLOR['yellow']}{char}{COLOR['reset']}")
        elif char == ':':
            result.append(f"{COLOR['blue']}{char}{COLOR['reset']}")
        elif char == ',':
            result.append(f"{COLOR['gray']}{char}{COLOR['reset']}")
        else:
            result.append(char)
        i += 1

    return ''.join(result)


def print_json(data: dict, truncate_content: int = 500):
    """打印带语法高亮的 JSON"""
    # 截断过长的 content
    if "content" in data and isinstance(data["content"], str) and len(data["content"]) > truncate_content:
        data = {**data, "content": data["content"][:truncate_content] + "...(truncated)"}
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    print(highlight_json(json_str))


def print_separator(char: str = "─", length: int = 60):
    """打印分割线"""
    print(f"{COLOR['gray']}{char * length}{COLOR['reset']}")


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
def agent_loop(messages: list):
    llm_with_tools = llm.bind_tools([bash])
    loop_count = 0

    while True:
        loop_count += 1

        # 打印分割线和 loop 计数
        if VERBOSE:
            print()
            print_separator()
            print(f"{COLOR['blue']}[LOOP {loop_count}]{COLOR['reset']}")
            print_separator()

        # 打印入参
        if VERBOSE:
            print(f"\n{COLOR['blue']}>>> REQUEST{COLOR['reset']}")
            for msg in messages:
                msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else {"type": type(msg).__name__}
                print_json(msg_dict)

        response = llm_with_tools.invoke(messages)

        # 打印出参
        if VERBOSE:
            print(f"\n{COLOR['green']}<<< RESPONSE{COLOR['reset']}")
            resp_dict = response.model_dump() if hasattr(response, "model_dump") else {"content": str(response)}
            print_json(resp_dict)

        messages.append(response)

        # If the model didn't call a tool, we're done
        if not response.tool_calls:
            if VERBOSE:
                print(f"\n{COLOR['gray']}[No tool calls, exiting loop]{COLOR['reset']}")
            return

        # Execute each tool call, collect results
        for tool_call in response.tool_calls:
            # 打印 tool 调用
            if VERBOSE:
                print(f"\n{COLOR['yellow']}>>> TOOL CALL: {tool_call['name']}{COLOR['reset']}")
                print_json(tool_call)

            # 执行命令
            print(f"\033[33m$ {tool_call['args']['command']}\033[0m")
            output = bash.invoke(tool_call["args"])

            # 打印 tool 结果
            if VERBOSE:
                print(f"\n{COLOR['yellow']}<<< TOOL RESULT{COLOR['reset']}")
                # 截断输出
                display_output = output[:1000] if len(output) > 1000 else output
                print(f"{COLOR['gray']}{display_output}{COLOR['reset']}")
            else:
                print(output[:200])

            messages.append(ToolMessage(
                content=output,
                tool_call_id=tool_call["id"],
            ))


if __name__ == "__main__":
    history = [SystemMessage(content=SYSTEM)]
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append(HumanMessage(content=query))
        agent_loop(history)
        # Print the final response
        last_msg = history[-1]
        if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
            print(last_msg.content)
        print()