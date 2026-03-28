# s02: Tool Use

`s01 > [ s02 ] s03 > s04 > s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12`

> *"ツールを足すなら、ハンドラーを1つ足すだけ"* -- ループは変わらない。新ツールは dispatch map に登録するだけ。
>
> **Harness 層**: ツール分配 -- モデルが届く範囲を広げる。

## 問題

`bash`だけでは、エージェントは何でもシェル経由で行う。`cat`は予測不能に切り詰め、`sed`は特殊文字で壊れ、すべてのbash呼び出しが制約のないセキュリティ面になる。`read_file`や`write_file`のような専用ツールなら、ツールレベルでパスのサンドボックス化を強制できる。

重要な点: ツールを追加してもループの変更は不要。

## 解決策

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

## 仕組み

1. 各ツールにハンドラ関数を定義する。パスのサンドボックス化でワークスペース外への脱出を防ぐ。

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

2. LangChainは`@tool`デコレータでツールを定義し、スキーマを自動生成する。

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

3. ループ内で名前によりハンドラをルックアップする。ループ本体はs01から不変。

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

ツール追加 = `@tool`関数を追加。ループは決して変わらない。

## s01からの変更点

| Component      | Before (s01)       | After (s02)                |
|----------------|--------------------|----------------------------|
| Tools          | 1 (bash only)      | 4 (bash, read, write, edit)|
| Dispatch       | Hardcoded bash call | `TOOL_DISPATCH` dict       |
| Path safety    | None               | `safe_path()` sandbox      |
| Agent loop     | Unchanged          | Unchanged                  |

## 試してみる

```sh
cd learn-claude-code
python agents/s02_tool_use.py
```

1. `Read the file requirements.txt`
2. `Create a file called greet.py with a greet(name) function`
3. `Edit greet.py to add a docstring to the function`
4. `Read greet.py to verify the edit worked`
