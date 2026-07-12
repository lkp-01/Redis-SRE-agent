# External MCP Client Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不改变现有 Redis 诊断主链的前提下，让当前 Agent 以 MCP Client 身份连接受信任配置中的外部 MCP Server，发现并调用显式允许的只读工具。

**Architecture:** 沿用 original 的 `MCPServerConfig -> MCPToolProvider -> ToolManager -> Agent tool loop` 形状，但按当前项目的短生命周期 CLI 边界做最小安全适配。每个 MCP Provider 的 transport/session 由当前轮 `ToolManager` 的 `AsyncExitStack` 独占并关闭；首期不恢复进程级单例连接池，也不开放 WRITE/UNKNOWN 工具。

**Tech Stack:** Python 3.12、MCP Python SDK、Pydantic Settings、LangChain Core、LangGraph、Click、pytest/pytest-asyncio。

---

## 执行方式

- 工作目录固定为 `D:\developer\redis_sre_prac\my_sre_agent`。
- `D:\developer\redis_sre_prac\original-redis-sre-agent-main` 永远只读。
- 从 Task 1 连续执行到 Task 7，阶段间不等待人工确认。
- 每个阶段先写失败测试，再做最小实现，再运行 focused tests。
- 任一相同失败最多进行 2 次基于错误日志的自修复；仍失败时立即停止并报告，不降低断言、不绕过 `ToolManager`。
- 不调用真实 OpenAI/DeepSeek，不连接公网 MCP；默认集成测试只启动本地 fake stdio MCP Server。
- 不打印或记录 header、token、env secret、认证 URL、Redis DSN 或工具敏感参数。

## 必须保持的主链

```text
Click query -> Thread create/recover -> router
-> ChatAgent / SRELangGraphAgent
-> existing StateGraph agent -> tools -> agent
-> ToolManager
   -> target discovery
   -> optional knowledge provider
   -> external MCP providers
   -> dynamically attached Redis providers
-> ResultEnvelope / ToolMessage
-> AgentResponse / Thread trace
```

以下接缝不得绕过：

- `ToolManager.resolve_tool_call()` 是所有外部 MCP 工具的唯一执行入口。
- `resolve_redis_targets -> attach_bound_targets -> RedisCommandToolProvider` 必须继续工作。
- Chat/Triage 的 StateGraph 拓扑、router、Thread 生命周期和 RAG readiness 逻辑不重写。
- MCP 未配置或不可用时，现有 Redis 诊断必须保持可用。

## 首期明确不做

- Agent 对外作为 MCP Server；不复制 `redis_sre_agent/cli/mcp.py` 或 MCP server package。
- original 的全局 `MCPConnectionPool`、API lifespan、worker/scheduler 生命周期。
- MCP WRITE/UNKNOWN 工具、人工审批、断点恢复。
- OAuth、token refresh、动态远端 catalog refresh、后台健康检查和自动重连。
- 图片 base64、音频、二进制 resource 注入 LLM。
- `langchain-mcp-adapters` 或另一套 ToolManager/ToolProvider 抽象。

## 安全契约

1. MCP Server 只能来自进程启动时的受信任 Settings；用户 query、LLM tool args 和 Thread context 都不能覆盖 server command/url/header/env。
2. `command` 与 `url` 必须且只能配置一个。
3. URL transport 仅接受 `sse` 或 `streamable_http`；默认要求 HTTPS，loopback HTTP 可用，其他 HTTP 需显式 `allow_insecure_http=true`。
4. 首期必须配置 `tools` allowlist；只有显式 `action_kind: read` 的工具可注册。
5. 远端 tool name、description、input schema 和 result 都视为不可信输入。
6. MCP tool 不得覆盖 target discovery、Redis diagnostics、knowledge 或另一个 MCP tool。
7. 内建工具在 LLM tool limit 中始终优先，MCP 工具最后裁剪。
8. `initialize`、`list_tools`、`call_tool` 和关闭流程必须有超时边界；取消信号不得被吞掉。

建议常量：

```text
MCP_MAX_DESCRIPTION_CHARS = 4000
MCP_MAX_SCHEMA_BYTES = 65536
MCP_MAX_RESULT_CHARS = 32000
MCP_TOOL_NAME_MAX_CHARS = 64
```

---

### Task 1: 冻结回归基线和 MCP 行为契约

**Files:**
- Create: `tests/unit/tools/mcp/__init__.py`
- Create: `tests/unit/tools/mcp/test_provider.py`
- Create: `tests/unit/tools/test_manager_mcp.py`
- Modify: `tests/unit/agent/test_stage5_end_to_end.py`

**Step 1: 记录实现前基线**

Run:

```powershell
python -m pytest -q
git status --short
```

Expected: 现有测试全部通过；若基线失败，先报告既有失败，不把它误归因于 MCP。

**Step 2: 写未配置 MCP 的失败测试**

断言：

- `mcp_servers={}` 时不导入/连接 MCP transport。
- target discovery、Redis provider 和 knowledge readiness 行为与实现前一致。
- `get_tools_for_llm()` 中不存在 `mcp_` 工具。

**Step 3: 写主链共存失败测试**

用 fake MCP provider 和 fake Redis client 断言同一 Manager 中同时存在：

- `resolve_redis_targets`
- MCP READ tool
- 动态绑定后的 Redis `info`

MCP 调用和 Redis 调用都必须经过 `ToolManager.resolve_tool_call()`。

**Step 4: 运行失败测试**

Run:

```powershell
python -m pytest -o addopts="" -q tests/unit/tools/mcp tests/unit/tools/test_manager_mcp.py tests/unit/agent/test_stage5_end_to_end.py
```

Expected: FAIL，原因是 MCP provider/package 和 Manager 加载实现尚不存在。

---

### Task 2: 增加 MCP SDK 和配置校验

**Files:**
- Modify: `pyproject.toml`
- Modify: `redis_sre_agent/core/config.py`
- Modify: `tests/unit/core/test_config.py`

**Original references:**
- `original-redis-sre-agent-main/redis_sre_agent/core/config.py::MCPToolConfig`
- `original-redis-sre-agent-main/redis_sre_agent/core/config.py::MCPServerConfig`
- `original-redis-sre-agent-main/pyproject.toml`

**Step 1: 写配置失败测试**

覆盖：

- command-only 和 URL-only 合法。
- command+url、两者皆空、非法 transport 拒绝。
- non-loopback 明文 HTTP 默认拒绝；显式 `allow_insecure_http=true` 才允许。
- `tools` 能解析 capability、description 和 action_kind。
- Settings/repr/validation error 不包含配置中的测试 secret。

**Step 2: 添加唯一新依赖**

在 `pyproject.toml` 添加：

```toml
"mcp>=1.23.3,<2.0.0",
```

不要引入 `langchain-mcp-adapters`。

**Step 3: 最小适配配置模型**

- 保留现有 `MCPToolConfig` 字段。
- 保留 `command/args/env/url/headers/transport/tools` 名称。
- 给 `MCPServerConfig` 增加 `allow_insecure_http: bool = False`。
- 用 Pydantic validator 实现 command/url、transport 和 URL scheme 校验。
- 不在 validator error 中插入 header/env 的实际值。

**Step 4: 安装并验证**

Run:

```powershell
python -m pip install -e .
python -m pytest -o addopts="" -q tests/unit/core/test_config.py
python -c "import mcp; print(mcp.__package__)"
```

Expected: 配置测试 PASS，输出 `mcp`，无真实连接。

**Step 5: Commit**

```powershell
git add pyproject.toml redis_sre_agent/core/config.py tests/unit/core/test_config.py
git commit -m "feat: validate external MCP client configuration"
```

---

### Task 3: 实现 turn-scoped MCPToolProvider

**Files:**
- Create: `redis_sre_agent/tools/mcp/__init__.py`
- Create: `redis_sre_agent/tools/mcp/provider.py`
- Modify: `tests/unit/tools/mcp/test_provider.py`

**Original references:**
- `original-redis-sre-agent-main/redis_sre_agent/tools/mcp/provider.py`
- `original-redis-sre-agent-main/tests/unit/tools/mcp_provider/test_mcp_provider.py`

**Step 1: 复制 Provider 架构形状**

保留：

- `MCPToolProvider(ToolProvider)`
- `_coerce_input_schema_dict()`
- `__aenter__()/__aexit__()`
- `_connect()/_disconnect()`
- stdio、SSE、Streamable HTTP transport 分支
- `initialize()`、`list_tools()`、`create_tool_schemas()`、`tools()`、`call_tool()`
- capability/description/action_kind override

删除或不复制：

- `get_active_mcp_runtime()` evaluation override。
- `MCPConnectionPool` 和 `use_pool` 分支。
- 记录 command/url/args、工具调用 args 或原始 exception 的日志。

**Step 2: 实现最小环境传递**

- stdio 子进程不复制完整 `os.environ`。
- 保留运行所需的 `PATH`，Windows 下保留 `SYSTEMROOT/COMSPEC/TEMP/TMP`。
- 只额外传递 server config 的 `env` 项。
- `${VAR}` 从父环境展开；未解析占位符时连接失败，但错误不含 secret value。

**Step 3: 实现确定性工具命名**

- 对 server/tool name 生成只包含 `[A-Za-z0-9_-]` 的 slug。
- 完整名称保持 original 风格：`mcp_<server>_<stable-hash>_<operation>`。
- 超过 64 字符时保留可读前缀并附加 hash。
- invoke closure 保存远端原始 tool name，`call_tool()` 不使用 slug。
- 同一 server 内重复原始名称或规范名冲突时整个 provider discovery 失败。

**Step 4: 实现只读 allowlist**

- `tools is None` 或空映射时，不暴露任何 MCP 工具并记录脱敏 warning。
- 远端未出现在 allowlist 的工具不暴露。
- allowlist 中非 `action_kind=read` 的工具不暴露。
- 不依赖 `UTILITIES` 的默认推断把 UNKNOWN 自动当成 READ。

**Step 5: 实现 schema/result 边界**

- description 截断到 4000 字符。
- input schema 转为普通 JSON dict，并限制到 64 KiB。
- 保留 `properties`、`required` 和 primitive/nullable 类型。
- structured content/text 总计最多 32000 字符。
- image/binary/resource 只返回类型和数量元数据，不传 base64。

**Step 6: 实现超时和安全错误**

- 使用 `settings.tool_timeout` 包裹 initialize/list/call。
- 不捕获 `asyncio.CancelledError`。
- 输出稳定错误码：`mcp_connect_failed`、`mcp_discovery_failed`、`mcp_timeout`、`mcp_tool_error`、`mcp_invalid_response`。
- 所有日志和返回错误先脱敏；不得返回 `str(exception)` 原文。

**Step 7: 完成 Provider 单元测试**

覆盖：三种 transport mock、filter/override、schema model coercion、确定性命名、重复名、超长内容、MCP `isError`、timeout、关闭、secret 不进入 caplog/result。

Run:

```powershell
python -m pytest -o addopts="" -q tests/unit/tools/mcp/test_provider.py
```

Expected: PASS，无子进程和网络残留。

**Step 8: Commit**

```powershell
git add redis_sre_agent/tools/mcp tests/unit/tools/mcp
git commit -m "feat: add turn-scoped MCP tool provider"
```

---

### Task 4: 在 ToolManager 注册和路由 MCP 工具

**Files:**
- Modify: `redis_sre_agent/tools/manager.py`
- Modify: `tests/unit/tools/test_manager_mcp.py`
- Modify: `tests/unit/tools/test_manager_stage3.py`

**Original references:**
- `original-redis-sre-agent-main/redis_sre_agent/tools/manager.py::_command_is_available`
- `original-redis-sre-agent-main/redis_sre_agent/tools/manager.py::_missing_local_mcp_arg_path`
- `original-redis-sre-agent-main/redis_sre_agent/tools/manager.py::_load_mcp_providers`

**Step 1: 复制启动前 guardrails**

- 复制并最小适配 `_command_is_available()`。
- 复制并最小适配 `_missing_local_mcp_arg_path()`。
- 缺少 command/本地入口只跳过对应 server，不产生 stack trace，不影响内建工具。

**Step 2: 填充现有 `_load_mcp_providers()` 插槽**

顺序必须保持：

```text
target discovery -> optional knowledge -> MCP -> target-scoped Redis providers
```

对每个 server：

1. 将 dict 转为 `MCPServerConfig`。
2. 构造 `MCPToolProvider`。
3. 通过 `self._stack.enter_async_context(provider)` 托管。
4. 读取候选 `Tool` 列表。
5. 应用 `exclude_mcp_categories`。
6. 在修改任何 Manager 表之前完成整组冲突预检。
7. 预检通过后才原子写入 `_routing_table/_tools/_tool_by_name/_providers/_loaded_provider_keys`。

任何 MCP 名称与现有工具或同批候选冲突时，跳过整个 provider，绝不覆盖。

**Step 3: 保证内建工具优先**

修改 `_llm_tool_priority()`：

- target discovery、Redis diagnostics、knowledge 和其他内建 provider 按现有顺序。
- `provider_name.startswith("mcp_")` 的所有工具排在内建工具之后。
- 超过 64 个工具时先裁剪 MCP。

**Step 4: 保持现有调用路径**

- `resolve_tool_call()` 继续调用 `tool.invoke()`。
- 不增加 Agent 到 MCP session 的直接引用。
- READ MCP 工具继续使用当前 per-turn call cache；不新增跨会话永久缓存。

**Step 5: 运行 Manager 测试**

Run:

```powershell
python -m pytest -o addopts="" -q tests/unit/tools/test_manager_mcp.py tests/unit/tools/test_manager_stage3.py
```

Expected: MCP 失败隔离、冲突 fail-closed、退出清理、内建优先全部 PASS。

**Step 6: Commit**

```powershell
git add redis_sre_agent/tools/manager.py tests/unit/tools/test_manager_mcp.py tests/unit/tools/test_manager_stage3.py
git commit -m "feat: register MCP providers through ToolManager"
```

---

### Task 5: 验证 Agent schema、动态工具集和证据链

**Files:**
- Modify only if failing tests require it: `redis_sre_agent/agent/helpers.py`
- Modify only if failing tests require it: `redis_sre_agent/agent/tool_execution.py`
- Modify: `tests/unit/agent/test_helpers_stage5.py`
- Modify: `tests/unit/agent/test_chat_agent_stage5.py`
- Modify: `tests/unit/agent/test_langgraph_agent_stage5.py`
- Modify: `tests/unit/agent/test_stage5_end_to_end.py`

**Step 1: 写 schema round-trip 测试**

验证 MCP JSON Schema 的 string/integer/number/boolean、nullable union、无 type optional 字段能经过 `build_adapters_for_tooldefs()` 传给 LLM。

**Step 2: 写 Chat/Triage 调用测试**

fake LLM 依次：

1. 选择 MCP READ tool。
2. 接收 MCP ToolMessage。
3. 继续选择 `resolve_redis_targets` 或 Redis `info`。
4. 返回最终文本。

断言 MCP 和 Redis 两类调用均进入顶层 `tool_envelopes`，错误状态不会被误判为 success。

**Step 3: 验证动态 target 重绑定**

MCP 工具初始加载后，`resolve_redis_targets` 导致 generation 增加；下一轮 `ensure_runtime_tools()` 必须同时保留 MCP 工具并加入 Redis provider 工具。

**Step 4: 最小修复原则**

- 如果现有 helpers 已通过，不修改生产代码。
- 只有真实测试证明 schema 或安全错误无法传递时才局部修改。
- 不改变 Chat/Triage graph edges、router、terminal synthesis 或 Thread 逻辑。

**Step 5: 运行 Agent focused tests**

Run:

```powershell
python -m pytest -o addopts="" -q tests/unit/agent/test_helpers_stage5.py tests/unit/agent/test_chat_agent_stage5.py tests/unit/agent/test_langgraph_agent_stage5.py tests/unit/agent/test_stage5_end_to_end.py
```

Expected: PASS；MCP 与 Redis 主链共存，所有工具仍经 ToolManager。

**Step 6: Commit**

```powershell
git add tests/unit/agent redis_sre_agent/agent/helpers.py redis_sre_agent/agent/tool_execution.py
git commit -m "test: verify MCP tools in agent diagnostic loop"
```

若两个生产文件未修改，只提交测试文件。

---

### Task 6: 本地 fake stdio MCP 集成验证

**Files:**
- Create: `tests/integration/fake_mcp_server.py`
- Create: `tests/integration/test_mcp_client_stdio.py`

**Step 1: 创建本地 fake MCP Server**

使用当前安装的 MCP SDK 构造 stdio server，至少暴露：

- `read_status`：显式只读，返回 text + structured content。
- `write_status`：用于证明客户端 allowlist/action gate 不会注册写工具。
- `large_result`：用于验证响应限制。
- `raise_secret_error`：错误包含仅测试使用的 sentinel secret，用于验证脱敏。

测试 server 不访问网络、不读取项目 `.env`、不打印 stderr secret。

**Step 2: 运行真实协议闭环**

通过临时 YAML 配置和当前 Python interpreter 启动 fake server，验证：

```text
subprocess start
-> MCP initialize
-> list_tools
-> ToolManager registration
-> call_tool(read_status)
-> ToolMessage/ResultEnvelope
-> ToolManager exit
-> subprocess/session/stream closed
```

**Step 3: 验证失败隔离和泄漏**

- fake server 不存在时，Redis/target tools 仍加载。
- timeout 后无遗留子进程。
- sentinel secret 不在 stdout/stderr/caplog/result。
- write tool 不在 Manager/LLM tool list。

**Step 4: 运行 integration test**

Run:

```powershell
python -m pytest -o addopts="" -q tests/integration/test_mcp_client_stdio.py
```

Expected: PASS；测试结束后没有 fake server 残留。

**Step 5: Commit**

```powershell
git add tests/integration/fake_mcp_server.py tests/integration/test_mcp_client_stdio.py
git commit -m "test: cover MCP stdio client end to end"
```

---

### Task 7: 文档、全量回归和最终审查

**Files:**
- Modify: `README.md`
- Modify: `.env.example` only if environment placeholders need documentation
- Create: `docs/codex/MCP_CLIENT.md`
- Modify: `docs/codex/ROADMAP.md`
- Modify: `docs/codex/CURRENT_STAGE.md`
- Modify: `docs/codex/STAGE_HISTORY.md`

**Step 1: 写中文使用文档**

记录：

- Stage 9 是外部 MCP Client 只读切片。
- YAML stdio 和 Streamable HTTP 示例，只使用 `${ENV_VAR}` 占位符。
- `tools` allowlist 和 `action_kind: read` 是必需安全边界。
- MCP 未配置/不可用不影响 Redis 诊断。
- 当前连接是 ToolManager turn-scoped，每轮重新发现工具。
- 非 loopback HTTP 的显式风险开关。
- 不支持写工具、进程池、OAuth 和 Agent-as-MCP-Server。

任何示例不得包含真实 token、密码、DSN、认证 URL 或本机 `.env` 内容。

**Step 2: 更新阶段记录**

- Roadmap 将外部 MCP Client 只读切片从“完整 MCP 生态”中单独标记为 Stage 9 已完成。
- “完整 MCP 生态”继续保留 pool、write approval、server/API/worker 等后续插槽。
- Stage History 记录从 original 复制/适配和有意不复制的部分。

**Step 3: 执行完整验证**

Run:

```powershell
python -m pip install -e .
python -m compileall redis_sre_agent tests
python -m pytest -q
git diff --check
git status --short
```

Expected:

- compileall PASS。
- 全量测试 PASS；只有既有或显式 live/integration skip。
- `git diff --check` 无错误。
- 变更仅位于 `my_sre_agent`。

**Step 4: 最终对抗性审查**

逐项确认：

- 未修改 original。
- 未新增第二套 tool routing。
- 未改变 Chat/Triage graph 和 target binding 主链。
- 未引入全局 MCP pool/background task。
- WRITE/UNKNOWN 工具无法注册或调用。
- MCP 工具不能覆盖内建工具。
- MCP 失败不阻断 Redis 诊断。
- transport/session/subprocess 均有所有者并关闭。
- 日志、异常、测试和文档中无 secret。
- 没有通过降低断言让测试变绿。

**Step 5: Commit**

```powershell
git add README.md .env.example docs/codex docs/plans/2026-07-12-mcp-client-integration.md
git commit -m "docs: document external MCP client slice"
```

如果 `.env.example` 未修改，不加入该文件。

## 停止条件

- 相同阻塞失败在原始失败和最多 2 次修复后仍存在。
- 实现必须修改 `original-redis-sre-agent-main`。
- 必须引入计划外大型依赖或重写 Agent/CLI 架构。
- 只能通过开放 WRITE 工具、绕过 ToolManager、关闭安全校验或泄露凭据才能继续。
- 当前 MCP SDK 与原 transport API 不兼容，且两次局部适配仍无法通过 fake stdio integration。

停止时保留工作区，报告：失败命令、脱敏后的错误、两次修复内容、已完成/未完成任务和下一起点。

## 完成定义

- `mcp_servers={}` 时现有行为零回归。
- 配置中的一个 MCP Server 可经 stdio 或 URL transport 完成 initialize/list/call。
- 仅显式 allowlist 中的 READ 工具进入 ToolManager 和 LLM tool list。
- 所有 MCP 调用经过 `ToolManager.resolve_tool_call()`。
- 名称冲突 fail-closed，内建 Redis/target/knowledge 工具优先。
- MCP Server 不可用时 Redis 诊断继续工作。
- 每轮 ToolManager 退出后 session、stream 和 stdio subprocess 被关闭。
- Chat/Triage 能执行 MCP 工具，并继续完成 target discovery/Redis diagnostics。
- 默认测试不访问公网、真实模型或真实外部 MCP Server。
- 全量测试、compileall 和 `git diff --check` 通过。
- README/Stage docs 清楚记录完成范围和后续插槽。
