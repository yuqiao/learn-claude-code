# s02: Tool Use (工具使用)

`s01 > [ s02 ] s03 > s04 > s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12`

> *"加一个工具, 只加一个 handler"* -- 循环不用动, 新工具注册进 dispatch map 就行。
>
> **Harness 层**: 工具分发 -- 扩展模型能触达的边界。

## 问题

只有 `bash` 时, 所有操作都走 shell。`cat` 截断不可预测, `sed` 遇到特殊字符就崩, 每次 bash 调用都是不受约束的安全面。专用工具 (`read_file`, `write_file`) 可以在工具层面做路径沙箱。

关键洞察: 加工具不需要改循环。

## 解决方案

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

## 工作原理

1. 每个工具有一个处理函数。路径沙箱防止逃逸工作区。

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

2. LangChain 使用 `@tool` decorator 定义工具，自动生成 schema。

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

3. 循环中按名称查找处理函数。循环体本身与 s01 完全一致。

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

加工具 = 加 `@tool` 函数。循环永远不变。

## 相对 s01 的变更

| 组件           | 之前 (s01)         | 之后 (s02)                     |
|----------------|--------------------|--------------------------------|
| Tools          | 1 (仅 bash)        | 4 (bash, read, write, edit)    |
| Dispatch       | 硬编码 bash 调用   | `TOOL_DISPATCH` 字典           |
| 路径安全       | 无                 | `safe_path()` 沙箱             |
| Agent loop     | 不变               | 不变                           |

## 试一试

```sh
cd learn-claude-code
python agents/s02_tool_use.py
```

试试这些 prompt (英文 prompt 对 LLM 效果更好, 也可以用中文):

1. `Read the file requirements.txt`
2. `Create a file called greet.py with a greet(name) function`
3. `Edit greet.py to add a docstring to the function`
4. `Read greet.py to verify the edit worked`
