#!/usr/bin/env python3
"""
verbose_callback.py - 可复用的 LangChain Callback Handler

提供详细的 API 调用日志，支持：
- LLM/Chat Model 请求/响应
- Tool 调用/结果
- JSON 语法高亮

使用方式：
    from verbose_callback import VerboseCallbackHandler

    # 创建 handler
    verbose = VerboseCallbackHandler()

    # 方式1：传递给 invoke
    response = llm.invoke(messages, config={"callbacks": [verbose]})

    # 方式2：绑定到 LLM
    llm_with_verbose = llm.with_config({"callbacks": [verbose]})

    # 方式3：全局设置
    from langchain_core.globals import set_debug
    set_debug(True)  # 使用内置调试模式
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


# ANSI 颜色码
COLOR = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "gray": "\033[90m",
    "blue": "\033[34m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "red": "\033[31m",
}


def highlight_json(text: str) -> str:
    """JSON 语法高亮"""
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

        # 处理数字
        if char.isdigit() or (char == '-' and i + 1 < len(text) and text[i + 1].isdigit()):
            j = i
            while j < len(text) and (text[j].isdigit() or text[j] in '.-eE+'):
                j += 1
            result.append(f"{COLOR['magenta']}{text[i:j]}{COLOR['reset']}")
            i = j
            continue

        # 处理关键字
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
            result.append(f"{COLOR['dim']}{char}{COLOR['reset']}")
        else:
            result.append(char)
        i += 1

    return ''.join(result)


def print_json(data: dict | list, truncate: int = 500) -> None:
    """打印带语法高亮的 JSON"""
    def truncate_strings(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: truncate_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [truncate_strings(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > truncate:
            return obj[:truncate] + "...(truncated)"
        return obj

    truncated = truncate_strings(data)
    json_str = json.dumps(truncated, ensure_ascii=False, indent=2)
    print(highlight_json(json_str))


def print_separator(char: str = "─", length: int = 60) -> None:
    """打印分割线"""
    print(f"{COLOR['gray']}{char * length}{COLOR['reset']}")


class VerboseCallbackHandler(BaseCallbackHandler):
    """
    详细日志 Callback Handler

    打印 LLM 请求/响应、Tool 调用等详细信息。

    Attributes:
        truncate_content: 内容截断长度
        truncate_output: 输出截断长度
        show_tokens: 是否显示 token 使用量
    """

    def __init__(
        self,
        *,
        truncate_content: int = 500,
        truncate_output: int = 1000,
        show_tokens: bool = True,
    ) -> None:
        self.truncate_content = truncate_content
        self.truncate_output = truncate_output
        self.show_tokens = show_tokens
        self._loop_count = 0

    def _print_header(self, title: str, color: str) -> None:
        """打印带颜色的标题"""
        print(f"\n{COLOR[color]}{COLOR['bold']}{title}{COLOR['reset']}")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """当 Chat Model 开始时调用"""
        self._loop_count += 1

        print()
        print_separator()
        print(f"{COLOR['blue']}{COLOR['bold']}[LLM CALL {self._loop_count}]{COLOR['reset']}")
        print_separator()

        self._print_header(">>> REQUEST (messages)", "blue")

        for msg_list in messages:
            for msg in msg_list:
                msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else {"type": type(msg).__name__}
                print_json(msg_dict, truncate=self.truncate_content)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """当 LLM 结束时调用"""
        self._print_header("<<< RESPONSE", "green")

        # 打印 token 使用量
        if self.show_tokens and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            if token_usage:
                print(f"\n{COLOR['dim']}Token Usage:{COLOR['reset']}")
                print(f"  {COLOR['cyan']}Input:{COLOR['reset']}  {token_usage.get('prompt_tokens', 'N/A')}")
                print(f"  {COLOR['cyan']}Output:{COLOR['reset']} {token_usage.get('completion_tokens', 'N/A')}")
                print(f"  {COLOR['cyan']}Total:{COLOR['reset']}  {token_usage.get('total_tokens', 'N/A')}")

        # 打印生成的内容
        for generation in response.generations:
            for gen in generation:
                if hasattr(gen, "message"):
                    # ChatGeneration
                    msg_dict = gen.message.model_dump() if hasattr(gen.message, "model_dump") else {"content": str(gen.message)}
                    print_json(msg_dict, truncate=self.truncate_content)
                elif hasattr(gen, "text"):
                    # Text generation
                    text = gen.text[:self.truncate_content]
                    if len(gen.text) > self.truncate_content:
                        text += "...(truncated)"
                    print(f"\n{COLOR['green']}{text}{COLOR['reset']}")

        # 检查是否有 tool calls
        if response.generations and response.generations[0]:
            gen = response.generations[0][0]
            if hasattr(gen, "message") and hasattr(gen.message, "tool_calls"):
                tool_calls = gen.message.tool_calls
                if not tool_calls:
                    print(f"\n{COLOR['dim']}[No tool calls, done]{COLOR['reset']}")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """当 LLM 出错时调用"""
        self._print_header("<<< ERROR", "red")
        print(f"{COLOR['red']}{error}{COLOR['reset']}")

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """当 Tool 开始时调用"""
        tool_name = serialized.get("name", "unknown")
        self._print_header(f">>> TOOL CALL: {tool_name}", "yellow")

        if inputs:
            print_json(inputs, truncate=self.truncate_content)

        # 对于 bash 工具，打印命令
        if inputs and "command" in inputs:
            print(f"\n{COLOR['yellow']}$ {inputs['command']}{COLOR['reset']}")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """当 Tool 结束时调用"""
        self._print_header("<<< TOOL RESULT", "yellow")

        output_str = str(output)
        if len(output_str) > self.truncate_output:
            output_str = output_str[:self.truncate_output] + "...(truncated)"

        print(f"{COLOR['dim']}{output_str}{COLOR['reset']}")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """当 Tool 出错时调用"""
        self._print_header("<<< TOOL ERROR", "red")
        print(f"{COLOR['red']}{error}{COLOR['reset']}")


# 便捷函数：创建全局 verbose handler
_verbose_handler: VerboseCallbackHandler | None = None


def get_verbose_handler(
    truncate_content: int = 500,
    truncate_output: int = 1000,
    show_tokens: bool = True,
) -> VerboseCallbackHandler:
    """获取或创建全局 verbose handler"""
    global _verbose_handler
    if _verbose_handler is None:
        _verbose_handler = VerboseCallbackHandler(
            truncate_content=truncate_content,
            truncate_output=truncate_output,
            show_tokens=show_tokens,
        )
    return _verbose_handler


def enable_verbose() -> VerboseCallbackHandler:
    """启用 verbose 模式，返回 handler 供使用"""
    return get_verbose_handler()