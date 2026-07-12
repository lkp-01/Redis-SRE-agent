# LLM Final Answer and Redis Thread Persistence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 恢复 original 的 LLM 最终回答链路与 Redis Thread 跨进程持久化，同时保持 Stage 5 target/tool 主链不变。

**Architecture:** Chat 保留 `agent -> tools -> agent -> END`，正常终态直接返回最后一条 AIMessage，只有迭代耗尽无文本才做 LLM terminal synthesis。Triage 保留 `agent -> tools -> agent -> reasoning -> END`，reasoning 按 evidence summary、TopicsList、Recommendation worker、Markdown composer 依次生成报告。ThreadManager 直接使用系统 Redis 的 List/Hash/String，不增加存储抽象。

**Tech Stack:** Python 3.12、LangChain Core、LangGraph、redis.asyncio、Pydantic、Click、pytest。

---

### Task 1: 锁定终态回答契约

**Files:**
- Modify: `tests/unit/agent/test_chat_agent_stage5.py`
- Modify: `tests/unit/agent/test_langgraph_agent_stage5.py`

**Steps:**
1. 添加 Chat fake LLM 工具循环后指定最终文本测试。
2. 添加 Chat 迭代上限 terminal synthesis 测试。
3. 添加 Triage TopicsList、Recommendation、composer 调用链测试。
4. 添加 composer 异常才进入确定性降级测试。
5. 运行 focused tests，确认旧实现失败。

### Task 2: 恢复 Chat 与 Triage LLM 终态链路

**Files:**
- Modify: `redis_sre_agent/agent/chat_agent.py`
- Modify: `redis_sre_agent/agent/langgraph_agent.py`
- Modify: `redis_sre_agent/agent/terminal_synthesis.py`
- Create: `redis_sre_agent/agent/subgraphs/recommendation_worker.py`
- Modify only if schema compatibility requires it: `redis_sre_agent/agent/models.py`

**Steps:**
1. 从 original 适配 Chat `_reached_iteration_limit`、`_synthesize_iteration_limit_response` 和正常终态返回顺序。
2. 从 original 适配 Triage `_summarize_envelopes_for_reasoning`、`_build_expand_evidence_tool`、`_compose_final_markdown`。
3. 从 original 裁剪 recommendation worker 的 StateGraph。
4. 用 original reasoning 主线替换确定性 reasoning node，并加入允许的 topic terminal synthesis。
5. 运行 Task 1 focused tests并修正最小兼容问题。

### Task 3: 锁定 Redis Thread 数据契约

**Files:**
- Create: `tests/unit/core/test_threads_redis.py`
- Modify: `tests/unit/core/test_redis.py`

**Steps:**
1. 添加两个 ThreadManager 实例共享 Redis 的创建、读取、追加、context、trace 测试。
2. 断言 messages 为 List，context/metadata 为 Hash，trace 为 JSON String。
3. 断言 Thread TTL 24 小时、trace TTL 7 天、消息与 Thread ID 是 26 位 ULID 形状。
4. 断言模块不存在 `_THREADS`、`_MESSAGE_TRACES`。
5. 运行 focused tests，确认旧实现失败。

### Task 4: 恢复 ThreadManager Redis backend

**Files:**
- Modify: `redis_sre_agent/core/threads.py`
- Modify: `redis_sre_agent/core/keys.py`
- Modify: `redis_sre_agent/core/redis.py`

**Steps:**
1. 补 `message_decision_trace`、`SRE_THREADS_SCHEMA`、`get_threads_index`。
2. 用标准库实现 ULID 同形 ID，不修改依赖。
3. 按 original 恢复 lazy Redis client、key 获取、完整 Thread 读写与 TTL。
4. 保留确定性 subject，避免引入无关 LLM 标题调用。
5. 运行 Task 3 focused tests。

### Task 5: 恢复 CLI 跨进程接线

**Files:**
- Modify: `redis_sre_agent/cli/query.py`
- Modify: `tests/unit/cli/test_main_stage3.py`
- Create: `tests/integration/test_cli_thread_redis_process.py`

**Steps:**
1. 使用 `get_redis_client()` 构造 ThreadManager。
2. 从 Thread 恢复 session/user/instance/cluster/context，只转换 user/assistant 历史。
3. Router 和 Agent 接收同一 conversation history。
4. Agent 成功后一次追加本轮 user/assistant，并单独保存 assistant trace。
5. 两个独立 Python 进程在同一系统 Redis 上验证历史、context、bindings、generation 与 trace。

### Task 6: 回归与交付

**Files:**
- Modify: `docs/codex/STAGE_HISTORY.md`

**Steps:**
1. 显式禁用真实外部 LLM key，运行 Agent/Thread/CLI focused tests。
2. 运行 target discovery、ToolManager、RedisCommandToolProvider、Stage 5 tests。
3. 运行 `python -m compileall redis_sre_agent tests`。
4. 运行 `python -m pytest -q`。
5. 记录 changed files、original 来源函数、刻意留白、验证结果和已知缺口。
