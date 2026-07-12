# DeepSeek LLM Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不改变 Stage 5 Redis 诊断主链的前提下，接入 DeepSeek `deepseek-v4-pro` / `deepseek-v4-flash`，支持模型级主副切换、保留 fake fallback，并提供可选的真实 API 验证。

**Architecture:** 从 original 最小适配 `core/llm_helpers.py` 作为唯一真实模型工厂。主 Agent 使用 Pro，并在一次模型调用失败时切换到 Flash；router、mini、nano 使用 Flash。Agent、StateGraph、ToolManager、target binding 和 Redis provider 控制流保持不变。

**Tech Stack:** Python 3.12、Pydantic Settings、LangChain Core、LangChain OpenAI、LangGraph、DeepSeek OpenAI-compatible API、pytest。

---

## 已确认的设计决策

- API 地址使用 `https://api.deepseek.com`。
- `OPENAI_MODEL=deepseek-v4-pro` 是主推理模型。
- `OPENAI_MODEL_MINI=deepseek-v4-flash` 同时作为主模型调用失败时的副模型。
- `OPENAI_MODEL_NANO=deepseek-v4-flash` 用于 router。
- 第一版显式关闭 DeepSeek thinking mode，避免工具调用轮次遗漏 `reasoning_content` 导致 400。
- 主副切换只重试失败的 LLM 调用，不重新启动整条 StateGraph，不主动重放已经完成的 Redis 工具。
- 未配置 API key 时使用 `FakeToolCallingLLM`；已经配置但主副模型都失败时返回已清洗的真实错误，不静默伪装成功。
- 单元测试不联网；真实 API 测试由 `RUN_DEEPSEEK_LIVE_TESTS=1` 显式开启。
- 不修改 read-only 的 `original-redis-sre-agent-main`。
- 工作区已有用户未提交改动，所有修改必须使用局部补丁，不覆盖或格式化无关代码，不自动提交 Git commit。

## Task 1：依赖和配置契约

**Files:**
- Modify: `pyproject.toml`
- Modify: `redis_sre_agent/core/config.py`
- Create: `.env.example`
- Modify: `tests/unit/core/test_config.py`

**Steps:**

1. 写失败测试，验证 DeepSeek 默认模型、LLM timeout、failover 开关和 thinking mode 配置。
2. 运行 `python -m pytest -q tests/unit/core/test_config.py`，确认新增断言先失败。
3. 按 original 版本边界加入 `langchain-openai>=1.2.1,<2.0.0`。
4. 在 `Settings` 中增加 `llm_timeout`、`llm_failover_enabled` 和 `deepseek_thinking_mode`，不改变现有 `SecretStr` 边界。
5. 创建不含真实密钥的 `.env.example`，写入 DeepSeek base URL 和 Pro/Flash 模型映射。
6. 重新安装 editable package 并运行配置测试。

## Task 2：最小 LLM 工厂与主副切换

**Files:**
- Create: `redis_sre_agent/core/llm_helpers.py`
- Create: `tests/unit/core/test_llm_helpers.py`

**Steps:**

1. 写失败测试覆盖 `create_llm()`、`create_mini_llm()`、`create_nano_llm()` 的 model/base URL/timeout/extra body。
2. 写失败测试覆盖主模型异常后只调用一次副模型，以及主副都失败时抛出清洗后的异常。
3. 从 original 适配三个公开 factory 名称和 `ChatOpenAI` 构造方式；只恢复当前 Stage 5 实际需要的 LangChain chat model 路径。
4. 实现轻量 `FailoverChatModel`：支持 `ainvoke()`、`invoke()`、`bind_tools()`；绑定工具时分别绑定主副模型并返回新的 failover wrapper。
5. `create_llm()` 返回 Pro + Flash wrapper；关闭 failover 时只返回 Pro。
6. mini/nano 直接创建 Flash，不递归嵌套 failover。
7. 日志只记录模型层级和异常类型，不记录 key、headers、请求体、Redis DSN 或模型输出全文。
8. 运行 `python -m pytest -q tests/unit/core/test_llm_helpers.py`。

## Task 3：接入 ChatAgent、Triage Agent 和 Router

**Files:**
- Modify: `redis_sre_agent/agent/chat_agent.py`
- Modify: `redis_sre_agent/agent/langgraph_agent.py`
- Modify: `redis_sre_agent/agent/router.py`
- Modify: `tests/unit/agent/test_chat_agent_stage5.py`
- Modify: `tests/unit/agent/test_langgraph_agent_stage5.py`
- Modify: `tests/unit/agent/test_router_stage5.py`

**Steps:**

1. 写失败测试，验证显式注入 LLM 的优先级最高。
2. 写失败测试，验证有 key 时 Agent 调用 `create_llm()`，无 key 时仍使用 fake。
3. 写失败测试，验证 router 有 key 时调用 `create_nano_llm()`，失败时使用确定性 fallback。
4. 对两个 Agent 构造器做最小修改，不改变公开参数、StateGraph 节点、ToolManager 或 provider 加载流程。
5. 对 router 做最小修改，保留 context/user_preferences 注入入口。
6. 清理 ChatAgent 全局缓存后运行三个 focused test 文件。

## Task 4：工具调用链和真实 DeepSeek 验证

**Files:**
- Create: `tests/integration/test_deepseek_live.py`
- Modify: `pyproject.toml`
- Reuse: `tests/unit/agent/test_stage5_end_to_end.py`

**Steps:**

1. 保留现有 fake E2E，验证 `resolve_redis_targets -> attach target -> Redis provider -> final response` 没有被绕过。
2. 增加 live test 标记；只有同时存在 DeepSeek key 和 `RUN_DEEPSEEK_LIVE_TESTS=1` 时运行。
3. live factory smoke test 验证 Pro/Flash 能返回内容，但不打印响应原文。
4. live tool-call smoke test绑定一个无副作用的本地工具，验证 DeepSeek 返回 tool call，并把 tool result 送回模型获得最终内容。
5. 如果本机没有 key，明确记录 `SKIPPED: missing credential`，不得伪造通过结果。
6. 如果本机有 key，运行 live tests；失败时最多按 AGENTS.md 自修复两次。

## Task 5：文档、回归验证和收尾记录

**Files:**
- Modify: `README.md`
- Modify: `docs/codex/CURRENT_STAGE.md`
- Modify: `docs/codex/STAGE_HISTORY.md`
- Modify: `docs/plans/2026-07-12-deepseek-llm-integration.md`

**Steps:**

1. 用中文说明 DeepSeek 配置、Pro/Flash 职责、failover 语义、fake fallback 和 live test 开关。
2. 明确 `.env` 不能提交，示例不含密钥。
3. 运行 `python -m compileall redis_sre_agent tests`。
4. 运行 focused LLM/Agent tests。
5. 运行 `python -m pytest -q` 完整测试套件。
6. 检查 `git diff --check` 和 `git status --short`，确认 original 未修改、用户已有改动未被覆盖。
7. 在本文件末尾填写实施结果、验证结果、已知缺口和未来功能。

## 验收标准

- 无 key：包可导入，CLI 和现有测试不联网，Agent 使用 fake。
- 有 key：ChatAgent/SRELangGraphAgent 默认使用 Pro，router 使用 Flash。
- Pro 单次调用异常：同一消息和工具绑定交给 Flash，不重新创建 Thread 或重跑完整图。
- 主副都失败：错误被清洗后向上抛出，不能返回 fake 成功报告。
- ToolManager、target discovery、Redis evidence 和现有 Stage 5 E2E 全部继续通过。
- 文档、异常、日志、测试结果不出现 API key、Redis 密码或连接串。

## 实施结果

- 状态：已完成代码实现和离线回归验证。
- 新增 `langchain-openai>=1.2.1,<2.0.0`，本机解析安装版本为 1.3.5。
- `Settings` 默认指向 DeepSeek，并增加 timeout、failover 和 thinking mode 配置。
- 新增 `core/llm_helpers.py`，恢复 original 的三个公开工厂名称。
- 新增单次调用级 `FailoverChatModel`，Pro 异常时切换 Flash；主副都失败时只暴露异常类型。
- ChatAgent、SRELangGraphAgent 和 router 已接入工厂，显式注入和无 key fake fallback 保持可用。
- 新增配置、工厂、Agent/router 和 live integration 测试。
- 更新 README、当前阶段和 Stage 历史。

### 验证记录

- 改动前：`python -m pytest -q`，67 passed。
- `tests/unit/core/test_config.py`：5 passed。
- `tests/unit/core/test_llm_helpers.py` + config：10 passed。
- Agent/router/Stage 5 E2E focused：18 passed。
- 完整套件：78 passed，2 skipped。
- live integration：2 skipped；本机未配置 DeepSeek key，`RUN_DEEPSEEK_LIVE_TESTS` 也未开启。
- 没有运行真实 Redis 或真实 DeepSeek 请求，没有输出任何密钥。

### 已知缺口

- 尚未取得真实 DeepSeek key，因此 Pro、Flash 和 tool-call round trip 的线上连通性仍待验证。
- failover 当前以“模型调用抛出异常”为条件，不包含响应质量评分或空响应切换。
- failover wrapper 只覆盖当前 Stage 5 使用的 `invoke`、`ainvoke`、`bind_tools`。
- live tool 测试使用无副作用本地函数，不是 live Redis 端到端测试。

## 未来需要补上的功能

- thinking mode 下 `reasoning_content` 在 LangChain/LangGraph 工具轮次中的无损回传。
- 对空响应、内容策略错误、模型不可用状态码的细粒度 failover 判定。
- 指数退避、熔断、provider 健康状态和恢复探测。
- 主副模型分别使用不同 API key/base URL 的多供应商容灾。
- token、延迟、费用和 failover 次数的可观测性。
- streaming、checkpoint/resume 和长会话上下文压缩。
- `with_structured_output`、stream/batch 等完整 LangChain Runnable 接口代理。
- DeepSeek strict tool schema 的兼容性审计。
- DeepSeek chat base URL 与未来 embedding provider/base URL 的配置拆分。
- 获得本地 key 后补跑真实 Pro/Flash 文本和工具调用测试。
- 在隔离的测试 Redis 上补充 DeepSeek → ToolManager → Redis provider 的真实端到端验证。
