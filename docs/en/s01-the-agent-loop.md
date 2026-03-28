# s01: The Agent Loop

`[ s01 ] s02 > s03 > s04 > s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12`

> *"One loop & Bash is all you need"* -- one tool + one loop = an agent.
>
> **Harness layer**: The loop -- the model's first connection to the real world.

## Problem

A language model can reason about code, but it can't *touch* the real world -- can't read files, run tests, or check errors. Without a loop, every tool call requires you to manually copy-paste results back. You become the loop.

## Solution

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

One exit condition controls the entire flow. The loop runs until the model stops calling tools.

## How It Works

1. User prompt becomes the first message.

```python
from langchain_core.messages import HumanMessage
messages.append(HumanMessage(content=query))
```

2. Send messages + tool definitions to the LLM.

```python
llm_with_tools = llm.bind_tools([bash])
response = llm_with_tools.invoke(messages)
```

3. Append the assistant response. Check for tool calls -- if none, we're done.

```python
messages.append(response)
if not response.tool_calls:
    return
```

4. Execute each tool call, collect results, append as ToolMessage. Loop back to step 2.

```python
from langchain_core.messages import ToolMessage
for tool_call in response.tool_calls:
    output = bash.invoke(tool_call["args"])
    messages.append(ToolMessage(
        content=output,
        tool_call_id=tool_call["id"],
    ))
```

Assembled into one function:

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def bash(command: str) -> str:
    """Run a shell command."""
    # ... implementation ...

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

That's the entire agent in under 30 lines. Everything else in this course layers on top -- without changing the loop.

## What Changed

| Component     | Before     | After                          |
|---------------|------------|--------------------------------|
| Agent loop    | (none)     | `while True` + `tool_calls`    |
| Tools         | (none)     | `@tool bash` (one tool)        |
| Messages      | (none)     | LangChain Message types        |
| Control flow  | (none)     | `not response.tool_calls`      |

## Try It

```sh
cd learn-claude-code
python agents/s01_agent_loop.py
```

1. `Create a file called hello.py that prints "Hello, World!"`
2. `List all Python files in this directory`
3. `What is the current git branch?`
4. `Create a directory called test_output and write 3 files in it`
