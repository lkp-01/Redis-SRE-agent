# Live Smoke Compatibility Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 live smoke 暴露的目标目录为空、DeepSeek structured output 不兼容和 Windows GBK JSON 输出失败。

**Architecture:** 保留 original 的资源 index 查询边界和 Triage structured-output 主线。当前 Redis 没有 Search module，因此在已有 `LightweightSearchIndex` 内提供小目录 SCAN fallback；DeepSeek structured output 改用 function calling；CLI 只在 stdout 编码无法表示字符时做 JSON 安全转义。

**Tech Stack:** Python 3.12、redis.asyncio、LangChain Core/OpenAI、Click、pytest。

---

### Task 1: Redis 小目录查询 fallback

**Files:**
- Modify: `redis_sre_agent/core/redis.py`
- Modify: `tests/unit/core/test_redis.py`

**Steps:**
1. 添加 CountQuery、FilterQuery、TAG 过滤、排序和分页失败测试。
2. 在 `LightweightSearchIndex.query()` 内使用 `scan_iter()` 和 `hgetall()` 读取 schema prefix。
3. 只解释当前 `core.redisearch` 生成的 AND TAG 表达式。
4. 运行实例、集群、target catalog focused tests。

### Task 2: DeepSeek structured output function calling

**Files:**
- Modify: `redis_sre_agent/agent/langgraph_agent.py`
- Modify: `redis_sre_agent/agent/subgraphs/recommendation_worker.py`
- Modify: `redis_sre_agent/agent/_compat.py`
- Modify: `tests/unit/agent/test_langgraph_agent_stage5.py`

**Steps:**
1. 让 fake structured LLM 记录并断言 `method="function_calling"`。
2. TopicsList 与 Recommendation 两处显式传入该 method。
3. 运行 Triage pipeline focused tests。
4. 用真实 mini LLM 做 TopicsList/Recommendation smoke。

### Task 3: Windows GBK JSON 输出

**Files:**
- Modify: `redis_sre_agent/cli/query.py`
- Modify: `tests/unit/cli/test_main_stage3.py`

**Steps:**
1. 添加 GBK 无法编码 emoji 时仍产生有效 JSON 的测试。
2. 增加编码感知 JSON render helper，并替换直接 `click.echo(json.dumps(...))`。
3. 运行 CLI 与 Thread tests。

### Task 4: 回归与 live smoke

**Files:**
- Modify: `docs/codex/STAGE_HISTORY.md`

**Steps:**
1. 运行 target discovery、ToolManager、Redis provider、Thread focused tests。
2. 运行 `python -m compileall redis_sre_agent tests` 和 `git diff --check`。
3. 隔离真实 LLM key运行全量单元测试。
4. 运行四条真实 CLI 主链路，确认目标发现能继续进入 Redis 工具。
5. 记录验证结果、SCAN fallback 限制和未恢复范围。
