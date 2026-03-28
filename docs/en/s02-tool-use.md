# s02: Tool Use

`s01 > [ s02 ] s03 > s04 > s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12`

> *"Adding a tool means adding one handler"* -- the loop stays the same; new tools register into the dispatch map.
>
> **Harness layer**: Tool dispatch -- expanding what the model can reach.

## Problem

With only `bash`, the agent shells out for everything. `cat` truncates unpredictably, `sed` fails on special characters, and every bash call is an unconstrained security surface. Dedicated tools like `read_file` and `write_file` let you enforce path sandboxing at the tool level.

The key insight: adding tools does not require changing the loop.

## Solution

```
+--------+      +-------+      +------------------+
|  User  | ---> |  LLM  | ---> | Tool Dispatch    |
| prompt |      |       |      | {                |
+--------+      +---+---+      |   bash: run_bash |
                    ^           |   read: run_read |
                    |           |   write: run_wr  |
                    +-----------+   edit: run_edit |
                    tool_result | }                |
                                +------------------+

The dispatch map is a dict: {tool_name: handler_function}.
One lookup replaces any if/elif chain.
```

## How It Works

1. Each tool gets a handler function. Path sandboxing prevents workspace escape.

```python
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_read(path: str, limit: int = None) -> str:
    text = safe_path(path).read_text()
    lines = text.splitlines()
    if limit and limit < len(lines):
        lines = lines[:limit]
    return "\n".join(lines)[:50000]
```

2. LangChain uses `@tool` decorator to define tools, auto-generating schema.

```python
from langchain_core.tools import tool

@tool
def bash(command: str) -> str:
    """Run a shell command."""
    return run_bash(command)

@tool
def read_file(path: str, limit: int = None) -> str:
    """Read file contents."""
    return run_read(path, limit)

@tool
def write_file(path: str, content: str) -> str:
    """Write content to file."""
    return run_write(path, content)

@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in file."""
    return run_edit(path, old_text, new_text)

TOOLS = [bash, read_file, write_file, edit_file]
TOOL_DISPATCH = {t.name: t for t in TOOLS}
```

3. In the loop, look up the handler by name. The loop body itself is unchanged from s01.

```python
for tool_call in response.tool_calls:
    handler = TOOL_DISPATCH.get(tool_call["name"])
    output = handler.invoke(tool_call["args"]) if handler \
        else f"Unknown tool: {tool_call['name']}"
    messages.append(ToolMessage(
        content=output,
        tool_call_id=tool_call["id"],
    ))
```

Add a tool = add an `@tool` function. The loop never changes.

## What Changed From s01

| Component      | Before (s01)       | After (s02)                |
|----------------|--------------------|----------------------------|
| Tools          | 1 (bash only)      | 4 (bash, read, write, edit)|
| Dispatch       | Hardcoded bash call | `TOOL_DISPATCH` dict       |
| Path safety    | None               | `safe_path()` sandbox      |
| Agent loop     | Unchanged          | Unchanged                  |

## Try It

```sh
cd learn-claude-code
python agents/s02_tool_use.py
```

1. `Read the file requirements.txt`
2. `Create a file called greet.py with a greet(name) function`
3. `Edit greet.py to add a docstring to the function`
4. `Read greet.py to verify the edit worked`
