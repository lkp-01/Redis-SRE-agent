# Redis SRE Agent 诊断切片

这是 `original-redis-sre-agent-main` 的裁剪复刻项目。Stage 8 在既有 Stage 5
诊断主链上恢复了可选的最小 RAG 闭环；Stage 9 又接入受信任配置中的外部 MCP Client
只读切片。两者都保留 original 的文件边界、ToolManager 路由和 StateGraph 控制流。

```text
本地 Markdown
-> chunk / hash / embedding
-> Redis Stack knowledge vector index
-> KnowledgeBaseToolProvider.search
-> ToolManager
-> Chat 或 Triage recommendation worker
-> 顶层 ResultEnvelope
-> AgentResponse.search_results
```

普通 Redis 诊断仍保持原来的链路：

```text
Click LazyGroup -> cli/query.py -> Redis Thread -> router
-> ChatAgent 或 SRELangGraphAgent -> StateGraph agent/tool loop
-> ToolManager -> target discovery / RedisCommandToolProvider
-> ResultEnvelope -> assistant response / message trace
```

## 外部 MCP Client

Stage 9 只允许配置 allowlist 中显式 `action_kind: read` 的 MCP 工具。stdio、SSE 和
Streamable HTTP transport 都由当前轮 `ToolManager` 独占；每轮退出时关闭 session、stream
和 stdio 子进程。MCP 未配置或连接失败不会阻断 target discovery、Redis diagnostics 或
可选 knowledge 工具。

```text
trusted Settings -> MCPServerConfig -> MCPToolProvider
-> ToolManager.resolve_tool_call()
-> ToolMessage -> ResultEnvelope
```

配置示例、安全边界、稳定错误码和当前不支持项见
[Stage 9 MCP Client 说明](docs/codex/MCP_CLIENT.md)。本阶段不支持 WRITE/UNKNOWN 工具、
全局连接池、OAuth、Agent-as-MCP-Server，也没有引入 `langchain-mcp-adapters`。

## RAG 三态

| 状态 | 条件 | Agent 可见的 knowledge search |
| --- | --- | --- |
| `disabled` | `RAG_ENABLED=false` | 不注册，也不检查 embedding、Redis Search 或 index |
| `not_ready` | 已启用，但 embedding 配置、Search/Vector、index 或 schema 未就绪 | 不注册；CLI/status 返回脱敏 reason code |
| `ready` | 独立 embedding 配置合法，Redis Search/Vector 可用，`sre_knowledge` 已存在且 schema 匹配 | 注册唯一的只读 search 工具 |

`redis-sre-agent status` 会显示当前状态和安全原因。普通诊断读取路径永远不会自动
创建或修改 knowledge index；只有显式 pipeline 摄取入口会调用
`ensure_knowledge_index(create_if_missing=true)`。

## Chat 与 embedding 配置隔离

DeepSeek chat 继续使用 `OPENAI_*` 名称，因为它提供 OpenAI-compatible chat API；
embedding 使用独立字段，两组凭据绝不回退或复用：

```dotenv
# Agent chat
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-pro
OPENAI_MODEL_MINI=deepseek-v4-flash

# RAG embedding
RAG_ENABLED=false
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
VECTOR_DIM=1536
VECTORIZER_FACTORY=
```

- `embedding_provider=openai` 必须配置独立 `EMBEDDING_API_KEY`。
- 自定义 `VECTORIZER_FACTORY` 可以不配置该 key，但返回对象必须实现异步
  `aembed()` 与 `aembed_many()`。
- 当前没有引入 sentence-transformers，因此不声明支持本地 HuggingFace provider。
- 默认测试只使用 fake vectorizer，不访问 DeepSeek、OpenAI 或其他 embedding 服务。

## 本地 Markdown 摄取

直接摄取 source tree：

```powershell
redis-sre-agent pipeline prepare-sources `
  --source-dir .\source_documents `
  --batch-date 2026-07-12 `
  --artifacts-path .\artifacts `
  --prepare-only

redis-sre-agent pipeline ingest `
  --batch-date 2026-07-12 `
  --artifacts-path .\artifacts
```

省略 `--prepare-only` 时，`prepare-sources` 会准备 artifact 后继续摄取同一批次。
也可以直接在 Python 中调用 `IngestionPipeline.ingest_source_documents()`。

摄取遵守以下一致性边界：

- 默认按 1000 字符、200 overlap 切分，优先在句号或空格边界结束。
- 先完成全部 chunk、content hash、embedding 和向量维度校验。
- 再用一个 Redis `MULTI/EXEC` 提交新 chunks、删除旧 chunks、更新文档 metadata
  和 source tracking。
- embedding 失败不会发出 Redis 写命令；事务未提交时旧知识仍可检索；提交结果未知时，
  相同输入使用确定性 key 重试即可恢复。
- R1b 只处理本地 Markdown/JSON artifact，不包含 skills、网页抓取、support tickets
  或多索引路由。

## 检索与 Agent 引用

独立检索：

```powershell
redis-sre-agent knowledge search "Redis memory pressure" --limit 5
```

Agent 查询仍使用同一个 ToolManager 路径：

```powershell
redis-sre-agent query "查找 Redis 内存压力的排查依据" --agent chat
redis-sre-agent query "deep triage redis memory" --agent triage
```

检索只使用 RedisVL `VectorRangeQuery` 或 `VectorQuery`，没有 SCAN 向量 fallback、
HybridQuery 或 RRF。实际 knowledge 调用会进入顶层 `tool_envelopes`，然后由
`AgentResponse` 从这些真实 envelope 派生 `search_results`。每条有效引用包含：

- `title`
- `source`
- `document_hash`
- `chunk_index`
- `score`
- `retrieval_kind=knowledge_search`

Triage recommendation worker 的内部 knowledge ToolMessage 会转换为 ResultEnvelope 并
合并回顶层；即使最终 composer 失败进入确定性报告，已执行的 knowledge evidence 也不会丢失。

## 运行与测试

```powershell
python -m pip install -e .
python -m compileall redis_sre_agent tests
python -m pytest -q
git diff --check
```

默认测试使用 fake LLM、fake vectorizer、fake transactional Redis/index 和本地 fake
stdio MCP server，不访问外部模型或公网 MCP Server。
真实 Redis Stack 集成必须显式指向隔离环境：

```powershell
$env:RUN_RAG_REDIS_INTEGRATION = "1"
$env:RAG_REDIS_TEST_URL = "<isolated Redis Stack URL>"
python -m pytest -o addopts="" -q -m integration tests/integration/test_rag_redis_stack.py
```

没有独立 embedding key 时，真实 embedding smoke 必须跳过；严禁用 DeepSeek chat key
代替。任何真实 key、Redis 密码、token、DSN 或带认证信息的连接串都不得写入文档、
日志、测试输出或提交记录。

## 有意保留的插槽

本阶段不恢复独立 Knowledge Agent、LLM ingest 工具、Hybrid/RRF/reranker、skills、
support tickets、knowledge pack、网页/PDF ingestion、MCP pool/write approval/server、
API、worker、scheduler、完整 eval、OpenTelemetry 或 UI。Click `LazyGroup`、Chat/Triage
StateGraph、Thread 持久化、target binding 和 Redis 只读诊断行为保持既有边界。
