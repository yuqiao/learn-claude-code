# s01: The Agent Loop

`[ s01 ] s02 > s03 > s04 > s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12`

> *"One loop & Bash is all you need"* -- 1つのツール + 1つのループ = エージェント。
>
> **Harness 層**: ループ -- モデルと現実世界を繋ぐ最初の接点。

## 問題

言語モデルはコードについて推論できるが、現実世界に触れられない。ファイルを読めず、テストを実行できず、エラーを確認できない。ループがなければ、ツール呼び出しのたびにユーザーが手動で結果をコピーペーストする必要がある。つまりユーザー自身がループになる。

## 解決策

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

1つの終了条件がフロー全体を制御する。モデルがツール呼び出しを止めるまでループが回り続ける。

## 仕組み

1. ユーザーのプロンプトが最初のメッセージになる。

```python
from langchain_core.messages import HumanMessage
messages.append(HumanMessage(content=query))
```

2. メッセージとツール定義をLLMに送信する。

```python
llm_with_tools = llm.bind_tools([bash])
response = llm_with_tools.invoke(messages)
```

3. アシスタントのレスポンスを追加し、ツール呼び出しを確認する。呼び出しがなければ終了。

```python
messages.append(response)
if not response.tool_calls:
    return
```

4. 各ツール呼び出しを実行し、結果を収集してToolMessageとして追加。ステップ2に戻る。

```python
from langchain_core.messages import ToolMessage
for tool_call in response.tool_calls:
    output = bash.invoke(tool_call["args"])
    messages.append(ToolMessage(
        content=output,
        tool_call_id=tool_call["id"],
    ))
```

1つの関数にまとめると:

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

これでエージェント全体が30行未満に収まる。本コースの残りはすべてこのループの上に積み重なる -- ループ自体は変わらない。

## 変更点

| Component     | Before     | After                          |
|---------------|------------|--------------------------------|
| Agent loop    | (none)     | `while True` + `tool_calls`    |
| Tools         | (none)     | `@tool bash` (one tool)        |
| Messages      | (none)     | LangChain Message types        |
| Control flow  | (none)     | `not response.tool_calls`      |

## 試してみる

```sh
cd learn-claude-code
python agents/s01_agent_loop.py
```

1. `Create a file called hello.py that prints "Hello, World!"`
2. `List all Python files in this directory`
3. `What is the current git branch?`
4. `Create a directory called test_output and write 3 files in it`
