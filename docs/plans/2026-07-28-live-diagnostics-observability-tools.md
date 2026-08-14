# Live Redis Diagnostics and Observability Tools Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在现有诊断主链上保留并验收 11 个 Redis 只读诊断工具，再依次加入 3 个 Prometheus 指标工具和 7 个 Loki 日志工具。

**Architecture:** 继续使用原项目的 `ToolProvider -> ToolManager -> Agent` 路由，不增加主机级编排层。Redis provider 保持当前实现；Prometheus 和 Loki provider 从只读参考项目最小复制、适配，并在目标绑定后按实例注册。所有单元测试必须 mock 外部连接，真实 Redis、Prometheus、Loki 只用于显式开启的集成 smoke。

**Tech Stack:** Python 3.12、redis-py、Pydantic Settings、prometheus-api-client、requests、httpx、pytest、pytest-asyncio。

---

## 1. 范围冻结

### 保留的工具

1. Redis Command：`info`、`slowlog`、`acl_log`、`config_get`、`client_list`、`cluster_info`、`replication_info`、`memory_stats`、`sample_keys`、`search_indexes`、`search_index_info`。
2. Prometheus：`query`、`query_range`、`search_metrics`。
3. Loki：`query`、`query_range`、`labels`、`label_values`、`series`、`volume`、`patterns`。

### 明确排除

- Host Telemetry 主机级诊断编排。
- Support Package 离线分析。
- Redis Cloud provider。
- Redis Enterprise Admin provider。
- 自动部署 Prometheus、Redis exporter、Loki 或日志采集器。
- Grafana dashboard、告警规则、Agent 自身 Prometheus instrumentation。
- Prometheus/Loki 写入、配置修改和任意命令执行。
- 新的认证体系、OAuth、审批流及 OpenAI 实网测试。

### 采用的交付方式

采用“Redis 基线 -> Prometheus -> Loki”的增量方式。不要把 Prometheus 和 Loki 放进同一个大改动，也不要先在仓库加入完整可观测性基础设施。这样每个 provider 都有独立测试门，外部服务未准备好时也不会阻塞普通 Redis 诊断。

## 2. 开始编码前由使用者准备的内容

### 现在即可准备：Redis

- 一个可访问的测试 Redis URL；不要把密码提交到仓库。
- 如果只验证主链，普通 standalone Redis 足够。
- 若希望 `search_indexes`、`search_index_info` 返回成功，需要带 Redis Search 模块的 Redis Stack；普通 Redis 应验证为结构化的“不支持”，不应让整个 Agent 失败。
- 若希望 `cluster_info` 返回成功，需要 Redis Cluster；standalone Redis 上的“不支持”同样是预期分支。
- 测试账号至少需要读取 INFO、SLOWLOG、ACL LOG、CONFIG GET、CLIENT LIST、CLUSTER INFO、MEMORY STATS、RANDOMKEY、TYPE、FT._LIST 和 FT.INFO 所需权限。生产权限应单独最小化，不在代码或文档中记录密码。

### Prometheus 阶段开始前

- 准备一个 Agent 能访问的 Prometheus HTTP API；Prometheus 是项目外服务，不嵌入 Python Agent。
- 如果要诊断 Redis 指标，需要运行 Redis exporter，并让 Prometheus 的 scrape target 显示为 `UP`。只启动 Prometheus 自身只能查到 Prometheus 自监控指标。
- 确认以下请求成功：`GET /-/ready`、`GET /api/v1/query?query=up`。
- 确认至少能查到一个 Redis 指标，例如环境所用 exporter 暴露的 `redis_up`。
- 准备 `TOOLS_PROMETHEUS_URL`；本切片不把 token 放进 URL，也不提前扩展认证协议。
- 当前环境记录显示 Docker CLI 存在但 daemon 未运行。若选择本地容器方案，应先启动 Docker daemon；也可以直接使用已有远程 Prometheus。

### Loki 阶段开始前

- 准备一个 Agent 能访问的 Loki HTTP API。
- Loki 必须已经收到 Redis 日志；仅运行空 Loki 没有诊断价值。
- 使用日志采集器把 Redis 日志写入 Loki。新环境优先使用 Grafana Alloy，不再新建 Promtail 配置。
- 预先确定稳定标签，至少建议有 `job="redis"` 和区分实例的 `instance` 或 `service`，并实际验证 `{job="redis"}` 能返回日志。
- 准备 `TOOLS_LOKI_URL`；多租户环境另准备 `TOOLS_LOKI_TENANT_ID`。
- `volume` 需要 Loki 开启 volume API；`patterns` 需要 pattern ingester。若暂不启用，这两个工具仍可注册，但集成测试应验证结构化错误，不得伪造成功。

## 3. 实施顺序

### Task 1：把现有 Redis Command provider 设为不可回退的基线

**Files:**

- Inspect: `redis_sre_agent/tools/diagnostics/redis_command/provider.py`
- Inspect: `redis_sre_agent/tools/manager.py`
- Test: `tests/unit/tools/test_redis_command_provider_stage4.py`
- Test: `tests/unit/tools/test_manager_stage3.py`

**Step 1: 验证现有 11 个 schema**

确认 schema 集合严格等于范围中的 11 个操作，不重新复制 provider，不改变现有控制流。

**Step 2: 运行现有基线测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/tools/test_redis_command_provider_stage4.py tests/unit/tools/test_manager_stage3.py -q
```

Expected: `11 passed`。

**Step 3: 只补真正缺失的回归测试**

如果当前测试尚未覆盖以下边界，才新增最小测试：

- Redis 客户端在第一次调用前不连接。
- 错误结果不暴露 Redis URL、密码或异常参数中的敏感内容。
- standalone 对 `cluster_info`、无 Search 模块对 `FT.*` 返回稳定的 unsupported/error envelope。
- `ToolManager` 可以解析带目标 hash 的全部 11 个工具名。

**Step 4: 运行完整单元测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit -q
```

Expected: 全部通过，且不访问 OpenAI 或外部观测服务。

**Step 5: Commit（由执行者在确认工作区现有改动后决定）**

```powershell
git add tests/unit/tools
git commit -m "test: lock redis diagnostic provider baseline"
```

### Task 2：先为 Prometheus 写失败的契约测试

**Files:**

- Create: `tests/unit/tools/metrics/__init__.py`
- Create: `tests/unit/tools/metrics/test_prometheus_provider.py`
- Create: `tests/unit/tools/metrics/test_prometheus_search_retry.py`

**Step 1: 从参考项目最小适配单元测试**

参考：

- `origina/tests/unit/tools/metrics/test_prometheus_provider.py`
- `origina/tests/unit/tools/metrics/test_prometheus_search_retry.py`

覆盖：

- 默认配置与 `TOOLS_PROMETHEUS_*` 环境变量。
- 3 个 schema 的名称、参数和 `METRICS` capability。
- lazy client。
- instant/range 查询的成功、空结果、无效时间和异常分支。
- `search_metrics` 的 pattern、label filter、重试分支。
- `tools()` 正确把 schema 绑定到 provider 方法。
- 所有网络/client 调用都 mock。

**Step 2: 运行并确认失败原因正确**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/tools/metrics -q
```

Expected: 因 `redis_sre_agent.tools.metrics.prometheus` 尚不存在而失败，而不是因真实网络连接失败。

### Task 3：最小复制 Prometheus provider

**Files:**

- Create: `redis_sre_agent/tools/metrics/__init__.py`
- Create: `redis_sre_agent/tools/metrics/prometheus/__init__.py`
- Create: `redis_sre_agent/tools/metrics/prometheus/provider.py`
- Modify: `pyproject.toml`

**Step 1: 添加直接依赖**

只添加 provider 源码直接使用的：

```toml
"prometheus-api-client>=0.6.0",
"requests>=2.31.0",
```

不要因为名称相似而添加 `prometheus-client`；该包用于应用暴露自身指标，不是查询 Prometheus API 的必需项。

**Step 2: 复制并最小适配 provider**

Source: `origina/redis_sre_agent/tools/metrics/prometheus/provider.py`

保持：

- `PrometheusConfig` 的 `TOOLS_PROMETHEUS_` 前缀。
- `query`、`query_range`、`search_metrics` 的 schema、核心控制流和结构化返回。
- lazy `PrometheusConnect` 客户端。
- 阻塞客户端调用通过线程边界执行，不能阻塞 Agent event loop。
- provider 错误只影响本次工具调用。

只允许适配当前裁剪版的 import、类型和错误脱敏接口；不要顺便重写为另一套 Prometheus 客户端。

**Step 3: 运行 provider 单元测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/tools/metrics -q
```

Expected: 全部通过，无真实 Prometheus 请求。

### Task 4：将 Prometheus 接入 ToolManager，但保持外部服务惰性

**Files:**

- Modify: `redis_sre_agent/core/config.py`
- Modify: `redis_sre_agent/tools/manager.py` only if existing generic provider loading cannot handle it
- Modify: `.env.example`
- Test: `tests/unit/tools/test_manager_stage3.py` or create `tests/unit/tools/test_manager_observability.py`

**Step 1: 写失败的 Manager 注册测试**

断言绑定 Redis target 后同时出现 11 个 `redis_command_*` 和 3 个 `prometheus_*` 工具；构造 Manager 不得访问 Prometheus。

**Step 2: 修改默认 provider 列表**

在 Redis provider 后追加：

```python
"redis_sre_agent.tools.metrics.prometheus.provider.PrometheusToolProvider"
```

不要增加 HostTelemetry provider。

**Step 3: 记录环境变量**

在 `.env.example` 只加入无密钥示例：

```dotenv
TOOLS_PROMETHEUS_URL=http://localhost:9090
TOOLS_PROMETHEUS_DISABLE_SSL=false
```

**Step 4: 运行 Manager 与 Agent 回归测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/tools tests/unit/agent -q
```

Expected: 全部通过；未运行 Prometheus 时，初始化仍成功。

### Task 5：Prometheus 显式集成 smoke

**Files:**

- Create: `tests/integration/tools/metrics/test_prometheus_provider.py`

**Step 1: 添加 opt-in 条件**

只有显式提供测试 URL 时才运行；默认测试套件应 skip，不能自动拉镜像或连接公网服务。

**Step 2: 先由使用者验证基础设施**

Run:

```powershell
Invoke-RestMethod "$env:TOOLS_PROMETHEUS_URL/-/ready"
Invoke-RestMethod "$env:TOOLS_PROMETHEUS_URL/api/v1/query?query=up"
```

Expected: Prometheus ready，查询响应的 `status` 为 `success`。

**Step 3: 运行集成测试**

覆盖 `query("up")`、短区间 `query_range`、`search_metrics`，再针对实际 Redis exporter 指标执行一次查询。

### Task 6：先为 Loki 写失败的契约测试

**Files:**

- Create: `tests/unit/tools/logs/__init__.py`
- Create: `tests/unit/tools/logs/loki/__init__.py`
- Create: `tests/unit/tools/logs/loki/test_loki_provider.py`

**Step 1: 从参考项目最小适配测试**

Source: `origina/tests/unit/tools/logs/loki/test_loki_provider.py`

第一组先覆盖 5 个基础工具：`query`、`query_range`、`labels`、`label_values`、`series`。第二组覆盖 `volume`、`patterns`。

必须覆盖：

- `TOOLS_LOKI_*` 配置和 tenant header。
- RFC3339、秒/毫秒/微秒/纳秒时间转换。
- 空 selector 的安全改写。
- 实例级 `prefer_streams` 和 `default_selector`。
- HTTP 4xx/5xx、非 JSON、连接失败的结构化错误。
- HTTP 调用完全 mock，不访问真实 Loki。

**Step 2: 运行并确认因模块缺失而失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/tools/logs -q
```

### Task 7：最小复制 Loki provider

**Files:**

- Create: `redis_sre_agent/tools/logs/__init__.py`
- Create: `redis_sre_agent/tools/logs/loki/__init__.py`
- Create: `redis_sre_agent/tools/logs/loki/provider.py`
- Modify: `pyproject.toml`

**Step 1: 添加直接依赖**

```toml
"httpx>=0.25.0",
```

即使当前虚拟环境因其他包间接安装了 httpx，也应把源码直接 import 的包声明为直接依赖。

**Step 2: 复制并最小适配 provider**

Source: `origina/redis_sre_agent/tools/logs/loki/provider.py`

保持 7 个 schema、时间解析、空 selector 修复、实例扩展配置和 HTTP API 路径。只适配裁剪项目已有协议与脱敏错误边界，不提前加入认证插件、重试框架或日志写入。

**Step 3: 运行 Loki 单元测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/tools/logs -q
```

Expected: 全部通过，无真实 Loki 请求。

### Task 8：注册 Loki 并执行分层 smoke

**Files:**

- Modify: `redis_sre_agent/core/config.py`
- Modify: `.env.example`
- Test: `tests/unit/tools/test_manager_observability.py`
- Create: `tests/integration/tools/logs/loki/test_loki_provider.py`

**Step 1: 在 provider 列表最后加入 Loki**

```python
"redis_sre_agent.tools.logs.loki.provider.LokiToolProvider"
```

最终顺序必须是 Redis Command、Prometheus、Loki；不要加入 HostTelemetry。

**Step 2: 增加无密钥环境示例**

```dotenv
TOOLS_LOKI_URL=http://localhost:3100
TOOLS_LOKI_TIMEOUT=30
TOOLS_LOKI_DEFAULT_SELECTOR={job="redis"}
```

tenant ID 只写变量名和空值，不提交真实租户信息。

**Step 3: Manager 单元测试**

断言每个目标共注册 21 个本范围内工具，并明确断言不存在 `host_telemetry`、`support_package`、`redis_cloud`、`re_admin` provider。

**Step 4: 基础 Loki live smoke**

先验证 `/ready`，再测试 `labels`、`label_values`、`series`、`query_range({job="redis"})`。这些通过后，才测试 instant `query`。

**Step 5: 可选能力 smoke**

- Loki 已启用 volume API：验证 `volume` 成功。
- Loki 已启用 pattern ingester：验证 `patterns` 成功。
- 未启用时：验证 provider 返回可理解的结构化错误，测试标记为环境能力缺失，而不是修改 Agent 假装支持。

### Task 9：最终主链验收与文档

**Files:**

- Modify: `README.md`
- Modify: `.env.example`
- Test: existing unit and opt-in integration suites

**Step 1: 文档只说明三类 provider**

写清楚：

- Agent 不负责部署 Prometheus/Loki。
- Prometheus 要先抓取 Redis exporter。
- Loki 要先摄取 Redis 日志并确定标签。
- 未配置外部服务时 Redis 直接诊断仍可运行；调用对应外部工具会返回结构化连接错误。
- 任何 URL、token、Redis 密码都不得出现在测试快照或日志里。

**Step 2: 编译与完整单元测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall -q redis_sre_agent
.\.venv\Scripts\python.exe -m pytest tests/unit tests/test_imports.py -q
```

Expected: 全部通过，不访问 OpenAI、Prometheus 或 Loki。

**Step 3: 分别运行可用的 opt-in 集成测试**

先 Redis，再 Prometheus，最后 Loki。某个外部服务未准备好时只跳过该服务的 smoke，不得跳过已有单元测试，也不得把服务不可用误报为实现成功。

## 4. 完成判定

- 每个绑定目标严格暴露本范围内 21 个工具：Redis 11、Prometheus 3、Loki 7。
- `ToolManager` 中不存在本次明确排除的四类 provider。
- provider 构造和 Agent 启动不主动连接 Prometheus/Loki。
- 所有外部错误都局部化并脱敏，不破坏 Redis 诊断主链。
- 默认测试不调用 OpenAI，也不要求 Redis、Prometheus、Loki 或 Docker。
- live smoke 只在使用者显式准备并开启后运行。

## 5. 已知限制

- Prometheus provider 沿用参考实现，目前没有专门的 bearer/basic-auth 配置；需要认证时先通过受控反向代理或另开安全设计任务，不能把密钥写入 URL。
- Loki provider 只提供 tenant header，没有在本切片增加通用认证插件。
- `volume`、`patterns` 的成功取决于 Loki 服务端功能开关。
- Redis Search 和 Cluster 工具的成功取决于目标形态；不支持时仍算“工具实现正确”，前提是返回稳定、可解释的错误。
- 当前工作区已有多项未提交改动及文档删除；实施时必须避开、保留这些用户改动，不能恢复被删除的旧 roadmap 或旧计划。
