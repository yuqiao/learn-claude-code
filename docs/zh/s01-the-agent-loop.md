# s01: The Agent Loop (智能体循环)

`[ s01 ] s02 > s03 > s04 > s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12`

> *"One loop & Bash is all you need"* -- 一个工具 + 一个循环 = 一个智能体。
>
> **Harness 层**: 循环 -- 模型与真实世界的第一道连接。

## 问题

语言模型能推理代码, 但碰不到真实世界 -- 不能读文件、跑测试、看报错。没有循环, 每次工具调用你都得手动把结果粘回去。你自己就是那个循环。

## 解决方案

```
+--------+      +-------+      +---------+
|  User  | ---> |  LLM  | ---> |  Tool   |
| prompt |      |       |      | execute |
+--------+      +---+---+      +----+----+
                    ^                |
                    |   tool_result  |
                    +----------------+
                    (loop until no tool_calls)
```

一个退出条件控制整个流程。循环持续运行, 直到模型不再调用工具。

## 工作原理

1. 用户 prompt 作为第一条消息。

```python
from langchain_core.messages import HumanMessage
messages.append(HumanMessage(content=query))
```

2. 将消息和工具定义一起发给 LLM。

```python
llm_with_tools = llm.bind_tools([bash])
response = llm_with_tools.invoke(messages)
```

3. 追加助手响应。检查是否有工具调用 -- 如果没有, 结束。

```python
messages.append(response)
if not response.tool_calls:
    return
```

4. 执行每个工具调用, 收集结果, 作为 ToolMessage 追加。回到第 2 步。

```python
from langchain_core.messages import ToolMessage
for tool_call in response.tool_calls:
    output = bash.invoke(tool_call["args"])
    messages.append(ToolMessage(
        content=output,
        tool_call_id=tool_call["id"],
    ))
```

组装为一个完整函数:

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def bash(command: str) -> str:
    """Run a shell command."""
    # ... 实现 ...

llm = ChatOpenAI(model=MODEL, base_url=BASE_URL)
llm_with_tools = llm.bind_tools([bash])

def agent_loop(messages: list):
    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return

        for tool_call in response.tool_calls:
            output = bash.invoke(tool_call["args"])
            messages.append(ToolMessage(
                content=output,
                tool_call_id=tool_call["id"],
            ))
```

不到 30 行, 这就是整个智能体。后面 11 个章节都在这个循环上叠加机制 -- 循环本身始终不变。

## 变更内容

| 组件          | 之前       | 之后                           |
|---------------|------------|--------------------------------|
| Agent loop    | (无)       | `while True` + `tool_calls`    |
| Tools         | (无)       | `@tool bash` (单一工具)        |
| Messages      | (无)       | LangChain Message 类型         |
| Control flow  | (无)       | `not response.tool_calls`      |

## 试一试

```sh
cd learn-claude-code
python agents/s01_agent_loop.py
```

试试这些 prompt (英文 prompt 对 LLM 效果更好, 也可以用中文):

1. `Create a file called hello.py that prints "Hello, World!"`
2. `List all Python files in this directory`
3. `What is the current git branch?`
4. `Create a directory called test_output and write 3 files in it`
