# s04: Subagents

`s01 > s02 > s03 > [ s04 ] s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12`

> *"Break big tasks down; each subtask gets a clean context"* -- subagents use independent messages[], keeping the main conversation clean.
>
> **Harness layer**: Context isolation -- protecting the model's clarity of thought.

## Problem

As the agent works, its messages array grows. Every file read, every bash output stays in context permanently. "What testing framework does this project use?" might require reading 5 files, but the parent only needs the answer: "pytest."

## Solution

```
Parent agent                     Subagent
+------------------+             +------------------+
| messages=[...]   |             | messages=[]      | <-- fresh
|                  |  dispatch   |                  |
| tool: task       | ----------> | while tool_use:  |
|   prompt="..."   |             |   call tools     |
|                  |  summary    |   append results |
|   result = "..." | <---------- | return last text |
+------------------+             +------------------+

Parent context stays clean. Subagent context is discarded.
```

## How It Works

1. The parent gets a `task` tool. The child gets all base tools except `task` (no recursive spawning).

```python
from langchain_core.tools import tool

@tool
def task(prompt: str) -> str:
    """Spawn a subagent with fresh context."""
    return run_subagent(prompt)
```

2. The subagent starts with `messages=[]` and runs its own loop. Only the final text returns to the parent.

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def run_subagent(prompt: str) -> str:
    sub_messages = [HumanMessage(content=prompt)]
    for _ in range(30):  # safety limit
        response = llm.invoke(sub_messages)
        sub_messages.append(response)
        if not response.tool_calls:
            break
        results = []
        for tool_call in response.tool_calls:
            handler = TOOL_HANDLERS.get(tool_call["name"])
            output = handler(**tool_call["args"])
            results.append(ToolMessage(
                tool_call_id=tool_call["id"],
                content=str(output)[:50000]))
        sub_messages.append(HumanMessage(content=results))
    return response.content or "(no summary)"
```

The child's entire message history (possibly 30+ tool calls) is discarded. The parent receives a one-paragraph summary as a normal `tool_result`.

## What Changed From s03

| Component      | Before (s03)     | After (s04)               |
|----------------|------------------|---------------------------|
| Tools          | 5                | 5 (base) + task (parent)  |
| Context        | Single shared    | Parent + child isolation  |
| Subagent       | None             | `run_subagent()` function |
| Return value   | N/A              | Summary text only         |

## Try It

```sh
cd learn-claude-code
python agents/s04_subagent.py
```

1. `Use a subtask to find what testing framework this project uses`
2. `Delegate: read all .py files and summarize what each one does`
3. `Use a task to create a new module, then verify it from here`
