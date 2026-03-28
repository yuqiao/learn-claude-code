# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 **Agent Harness 教学项目**，教授如何构建 AI Agent 的运行环境（harness），而非 Agent 本身。核心理念：

> **The model is the agent. The code is the harness.**

项目包含 12 个渐进式 session（s01-s12），每个 session 添加一个 harness 机制。

## 常用命令

### Python Agents
```bash
pip install -r requirements.txt
cp .env.example .env  # 配置 ANTHROPIC_API_KEY 和 MODEL_ID

python agents/s01_agent_loop.py       # 入门：单个 agent loop
python agents/s12_worktree_task_isolation.py  # 完整功能
python agents/s_full.py               # 所有机制整合版
```

### Web 平台
```bash
cd web && npm install && npm run dev  # http://localhost:3000
```

### CI
```bash
cd web && npx tsc --noEmit && npm run build
```

## 架构

```
learn-claude-code/
├── agents/           # Python 参考实现 (s01-s12 + s_full)
├── docs/{en,zh,ja}/  # 三语文档
├── web/              # Next.js 交互学习平台
├── skills/           # Skill 文件 (s05)
└── .github/workflows/ # CI: TypeScript 类型检查 + 构建
```

## 核心模式

**Agent Loop**（所有 session 的基础）：
```python
def agent_loop(messages):
    while True:
        response = client.messages.create(model, messages, tools)
        messages.append(response)
        if response.stop_reason != "tool_use":
            return
        results = [execute_tool(block) for block in response.content]
        messages.append({"role": "user", "content": results})
```

**12 个 Harness 机制**：
| Session | 机制 | 格言 |
|---------|------|------|
| s01 | Agent Loop | One loop & Bash is all you need |
| s02 | Tool Dispatch | Adding a tool means adding one handler |
| s03 | TodoWrite | An agent without a plan drifts |
| s04 | Subagent | Break big tasks down; each subtask gets a clean context |
| s05 | Skill Loading | Load knowledge when you need it, not upfront |
| s06 | Context Compact | Context will fill up; you need a way to make room |
| s07 | Task System | Break big goals into small tasks, order them, persist to disk |
| s08 | Background Tasks | Run slow operations in the background |
| s09 | Agent Teams | When the task is too big for one, delegate to teammates |
| s10 | Team Protocols | Teammates need shared communication rules |
| s11 | Autonomous Agents | Teammates scan the board and claim tasks themselves |
| s12 | Worktree Isolation | Each works in its own directory, no interference |

## 环境变量

```bash
ANTHROPIC_API_KEY=sk-ant-xxx    # 必需
MODEL_ID=claude-sonnet-4-6       # 必需
ANTHROPIC_BASE_URL=...          # 可选，用于兼容提供商
```

支持 Anthropic 兼容提供商：MiniMax、GLM、Kimi、DeepSeek。

## Skill 文件格式

```markdown
---
name: skill-name
description: One-line description
---

Skill body content loaded via tool_result.
```

## 项目范围

这是一个 0→1 学习项目，有意简化或省略了：
- 完整的事件/钩子系统
- 基于规则的权限治理
- Session 生命周期控制
- 完整的 MCP 运行时细节