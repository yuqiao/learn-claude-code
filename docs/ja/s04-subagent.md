# s04: Subagents

`s01 > s02 > s03 > [ s04 ] s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12`

> *"大きなタスクを分割し、各サブタスクにクリーンなコンテキストを"* -- サブエージェントは独立した messages[] を使い、メイン会話を汚さない。
>
> **Harness 層**: コンテキスト隔離 -- モデルの思考の明晰さを守る。

## 問題

エージェントが作業するにつれ、messages配列は膨張し続ける。すべてのファイル読み取り、すべてのbash出力がコンテキストに永久に残る。「このプロジェクトはどのテストフレームワークを使っているか」という質問は5つのファイルを読む必要があるかもしれないが、親に必要なのは「pytest」という答えだけだ。

## 解決策

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

## 仕組み

1. 親に`task`ツールを追加する。子は`task`を除くすべての基本ツールを取得する(再帰的な生成は不可)。

```python
from langchain_core.tools import tool

@tool
def task(prompt: str) -> str:
    """Spawn a subagent with fresh context."""
    return run_subagent(prompt)
```

2. サブエージェントは`messages=[]`で開始し、自身のループを実行する。最終テキストだけが親に返る。

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def run_subagent(prompt: str) -> str:
    sub_messages = [HumanMessage(content=prompt)]
    for _ in range(30):  # safety limit
        response = llm.invoke(sub_messages, config={"system": SUBAGENT_SYSTEM})
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
        sub_messages.extend(results)
    return "".join(
        block.text for block in response.content
        if hasattr(block, "text") and block.text
    ) or "(no summary)"
```

子のメッセージ履歴全体(30回以上のツール呼び出し)は破棄される。親は1段落の要約を通常の`tool_result`として受け取る。

## s03からの変更点

| Component      | Before (s03)     | After (s04)               |
|----------------|------------------|---------------------------|
| Tools          | 5                | 5 (base) + task (parent)  |
| Context        | Single shared    | Parent + child isolation  |
| Subagent       | None             | `run_subagent()` function |
| Return value   | N/A              | Summary text only         |

## 試してみる

```sh
cd learn-claude-code
python agents/s04_subagent.py
```

1. `Use a subtask to find what testing framework this project uses`
2. `Delegate: read all .py files and summarize what each one does`
3. `Use a task to create a new module, then verify it from here`