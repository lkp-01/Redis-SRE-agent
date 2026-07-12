# RAG Minimum Closed Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在现有 Stage 5 Redis 诊断主链上，按 original 文件边界恢复“本地 Markdown 摄取 → 文档切分 → embedding → Redis 向量索引 → knowledge 工具检索 → Agent 调用 → 结构化引用输出”的最小闭环。

**Architecture:** RAG 是显式启用的可选能力，只有通过配置、Redis Search/Vector 和 knowledge index 就绪检查后，`ToolManager` 才向 LLM 注册 knowledge search。写路径通过精简 ingestion pipeline 原子替换文档，读路径复用当前 `ToolManager`、Agent tool loop、`ResultEnvelope` 和 `AgentResponse.search_results`；Triage recommendation worker 的内部知识工具证据必须合并回顶层 `signals_envelopes`。

**Tech Stack:** Python 3.12、Redis/Redis Stack、RedisVL、Pydantic Settings、LangChain Core、LangGraph、Click、pytest。

---

## 对抗性审查处理结果

以下六项建议全部接受，并已落实为实施契约：

1. 增加 `rag_enabled` 和 disabled/not-ready/ready 三态控制。
2. embedding key/base URL 与 DeepSeek chat 配置完全分离。
3. 摄取拆为 R1a/R1b，禁止复制 `pipeline_workflow_mixin.py` 的 skills 隐藏依赖。
4. recommendation worker 的 knowledge ToolMessage 必须回传为顶层 ResultEnvelope。
5. 明确 `ensure_knowledge_index()` 的创建责任，并保留 `product_labels` 字段。
6. 文档更新先完成全部 embedding，再在单个 Redis 原子提交中替换，失败时保留旧版本或可确定性重试。

## 执行边界

- 只修改 `D:\developer\redis_sre_prac\my_sre_agent`。
- `D:\developer\redis_sre_prac\original-redis-sre-agent-main` 永远只读，只用于复制和对照。
- 优先复制 original 的函数、类、字段、key 和文件边界；仅在本计划明确指出的冲突处做适配。
- 不改写 Chat/Triage 的 StateGraph 主控制流，不绕过 `ToolManager`。
- 默认测试不得调用真实 OpenAI、DeepSeek 或 embedding API。
- 当前普通 Redis 没有 RediSearch module；不得用 `LightweightSearchIndex` 的 SCAN fallback 冒充向量检索。
- 普通诊断在 `rag_enabled=false` 时不连接 RAG、不加载 knowledge provider、不暴露 knowledge tool。
- RAG 已启用但未就绪时，不向 LLM 暴露 knowledge tool；status/knowledge/pipeline 入口必须给出脱敏后的明确原因。
- 真实 RAG 需要带 Search/Vector 能力的 Redis；缺失时不能伪装成“成功但无结果”。

## 本阶段非目标

- 独立 `KnowledgeOnlyAgent`。
- Agent 可调用的 ingest/write 工具。
- Web scraper、Redis Docs Git 子模块、PDF/Notebook/Cloud API ingestion。
- HybridQuery、精确短语检索、RRF、reranker。
- skills、support tickets、pinned startup context。
- knowledge pack、API、worker、scheduler、evaluation suite、UI。

## 必须保持的数据契约

```text
Index: sre_knowledge
Key:   sre_knowledge:{document_hash}:chunk:{chunk_index}
Meta:  sre_knowledge_meta:{document_hash}

Required fields:
id, document_hash, content_hash, title, content, source, category,
doc_type, name, summary, priority, pinned, severity,
product_labels, product_label_tags, version, chunk_index, created_at, vector

Vector field:
algorithm=flat, datatype=float32, distance_metric=cosine,
dims=settings.vector_dim

Search result:
id, document_hash, chunk_index, title, content, source, category,
doc_type, name, summary, priority, pinned, version, score
```

## RAG 状态契约

```text
disabled:
  rag_enabled=false
  不检查 Redis/embedding，不注册 KnowledgeBaseToolProvider

not_ready:
  rag_enabled=true，但 embedding 配置无效、Redis 无 Search/Vector、
  knowledge index 缺失或 schema 不兼容
  不注册 knowledge tool；状态/摄取/检索入口返回明确脱敏错误

ready:
  embedding 配置合法，Redis Search/Vector 可用，knowledge index 已存在且 schema 可用
  ToolManager 才注册 KnowledgeBaseToolProvider.search
```

正常诊断读取就绪状态时不得自动创建或修改索引。只有显式摄取/index 管理入口可以创建索引。

---

### Task 1 / R0: RAG 开关、独立 embedding 配置、索引责任与 RedisVL 基础

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `redis_sre_agent/core/config.py`
- Modify: `redis_sre_agent/core/keys.py`
- Modify: `redis_sre_agent/core/redis.py`
- Create: `redis_sre_agent/core/vectorizer_helpers.py`
- Modify: `redis_sre_agent/tools/manager.py`
- Modify: `redis_sre_agent/cli/main.py`
- Create: `tests/unit/core/test_vectorizer_helpers.py`
- Create: `tests/unit/core/test_rag_readiness.py`
- Modify: `tests/unit/core/test_redis.py`
- Modify: `tests/unit/core/test_config.py`
- Modify: `tests/unit/tools/test_manager_stage3.py`
- Create: `tests/integration/test_rag_redis_stack.py`

**Original references:**
- `original-redis-sre-agent-main/redis_sre_agent/core/vectorizer_helpers.py`
- `original-redis-sre-agent-main/redis_sre_agent/core/redis.py::_build_document_schema`
- `original-redis-sre-agent-main/redis_sre_agent/core/redis.py::get_vectorizer`
- `original-redis-sre-agent-main/redis_sre_agent/core/redis.py::get_knowledge_index`
- `original-redis-sre-agent-main/redis_sre_agent/core/keys.py`

**Implementation contract:**

```text
rag_enabled: bool = false
embedding_api_key: Optional[SecretStr] = None
embedding_base_url: Optional[str] = None
embedding_provider: str
embedding_model: str
vector_dim: int
vectorizer_factory: Optional[str]
```

- Default vectorizer factory只能读取 `embedding_*` 字段。
- 禁止回退到 `openai_api_key` 或 `openai_base_url`。
- `embedding_provider=openai` 时必须有独立 `embedding_api_key`；自定义 factory 可不需要该 key。
- 本阶段不声明支持 local sentence-transformers，除非用户另行批准对应依赖。

**Steps:**

1. 写失败测试，固定 `rag_enabled=false` 时普通 CLI/ToolManager 不构造 vectorizer、不访问 Redis RAG index、不加载 knowledge provider。
2. 写失败测试，固定三态 readiness 及安全 reason code：`disabled`、`embedding_config_invalid`、`redis_search_unavailable`、`index_missing`、`schema_mismatch`、`ready`。
3. 写失败测试，证明 chat 的 `openai_api_key/openai_base_url` 即使已配置，也绝不会传给 vectorizer factory。
4. 在 `Settings` 增加 `rag_enabled`、`embedding_api_key`、`embedding_base_url`；保留已有 provider/model/dim/cache/factory 字段。所有 secret 使用 `SecretStr`。
5. 在依赖中只加入本闭环必需的 RedisVL；不要加入 sentence-transformers、text-splitters、scraper/PDF 依赖。
6. 从 original 复制并最小适配 `vectorizer_helpers.py`，保留 `Vectorizer`、`VectorizerFactory`、`set_vectorizer_factory()` 和 async API 校验。
7. 在 `core/redis.py` 增加 original 兼容的 knowledge schema，包含 `product_labels` 和 `product_label_tags`。
8. 实现只构造对象的 `get_knowledge_index()`；它不得隐式创建索引。
9. 实现显式 `ensure_knowledge_index(config, create_if_missing)`：`rag_enabled=false` 时拒绝执行；启用后检查 `FT.SEARCH/FT.CREATE` 能力、检查 index/schema、在 `create_if_missing=true` 时创建、创建后验证 exists/query 能力。
10. 实现 RAG readiness helper。普通诊断和 ToolManager 只用 `create_if_missing=false`；pipeline ingest 使用 `true`。
11. 将 knowledge provider 从静态 `_always_on_providers` 中移出。ToolManager 始终加载 target discovery，但仅在 `rag_enabled=true` 且 readiness=ready 时加载 knowledge provider。
12. enabled/not-ready 时 ToolManager 不注册 knowledge tool，并保存/记录脱敏状态；普通 Redis 诊断仍可继续。`redis-sre-agent status` 显示 RAG 三态和安全原因。
13. 补齐 original 的 knowledge key helpers。
14. 对 vector 维度做写入前校验；模型、维度或 schema 不一致时要求显式重建。
15. 写 opt-in Redis Stack integration：`FT.CREATE → index.load/Hash write → VectorQuery → source/score assertion`，让索引创建问题在 R0 暴露。
16. 运行：

```powershell
python -m pytest -o addopts="" -q tests/unit/core/test_vectorizer_helpers.py tests/unit/core/test_rag_readiness.py tests/unit/core/test_redis.py tests/unit/core/test_config.py tests/unit/tools/test_manager_stage3.py
python -m compileall redis_sre_agent tests
```

Expected: disabled 普通诊断完全不见 knowledge tool；enabled/not-ready 不暴露工具且原因明确；所有测试无网络请求、无 secret/Redis URL 输出。

---

### Task 2 / R1a: 直接本地 Markdown → Chunk → Vector Index

**Files:**
- Create: `redis_sre_agent/pipelines/__init__.py`
- Create: `redis_sre_agent/pipelines/scraper/__init__.py`
- Create: `redis_sre_agent/pipelines/scraper/base.py`
- Create: `redis_sre_agent/pipelines/ingestion/__init__.py`
- Create: `redis_sre_agent/pipelines/ingestion/document_processor.py`
- Create: `redis_sre_agent/pipelines/ingestion/deduplication.py`
- Create: `redis_sre_agent/pipelines/ingestion/processor_source_helpers.py`
- Create: `redis_sre_agent/pipelines/ingestion/processor_indexing_helpers.py`
- Create: `redis_sre_agent/pipelines/ingestion/_processor_impl.py`
- Create: `redis_sre_agent/pipelines/ingestion/processor.py`
- Create: `tests/unit/pipelines/ingestion/test_document_processor.py`
- Create: `tests/unit/pipelines/ingestion/test_deduplication.py`
- Create: `tests/unit/pipelines/ingestion/test_ingestion_processor.py`

**Original references:** 对应文件从 `original-redis-sre-agent-main/redis_sre_agent/pipelines/` 复制后，只保留 knowledge/local Markdown 路径。

**Explicit cuts:**

- R1a 不创建、不导入 `pipeline_workflow_mixin.py`。
- 不导入 `redis_sre_agent.skills.discovery`。
- 不创建 skills/support-ticket index，不做多索引路由。
- 不恢复 scraper orchestrator、网页抓取、并行 batch、stale-scope cleanup。

**Steps:**

1. 写 `ScrapedDocument`、`document_hash` 和 Markdown metadata/source identity 稳定性失败测试。
2. 从 original `scraper/base.py` 只复制 enums 与 `ScrapedDocument`；`ArtifactStorage` 留到 R1b。
3. 写 chunk 失败测试：默认 1000 字符、200 overlap、最小 chunk、句号/空格边界、front matter 剥离、短文档整篇保留。
4. 复制 `DocumentProcessor` 和必要 source helpers，不引入 LangChain text splitter。
5. 写更新一致性失败测试：embedding 失败时零 Redis mutation；写入失败时旧 source pointer/旧 chunks 仍可检索或同一输入可确定性重试恢复。
6. 裁剪 `DocumentDeduplicator`：只保留 deterministic keys、knowledge metadata/source tracking、现有 chunk 读取和单文档 replacement。
7. 在任何 Redis 删除/覆盖前，完成全部 chunk 构造、content hash、embedding 和维度校验。
8. 将新 chunk hash 写入、旧多余 chunk 删除、document metadata 和 source tracking 更新放在单个 Redis `MULTI/EXEC` 原子提交中。若 RedisVL `index.load()` 不能满足该边界，使用 index 的底层 client transaction 写 Hash；RediSearch 仍自动索引同一 schema。
9. 原子提交前不得删除旧版本；提交失败不得提前更新 source tracking。所有命令输入预先验证，使重试幂等。
10. 实现精简 `IngestionPipeline.ingest_source_documents(source_dir)`：要求 `rag_enabled=true`，先 `ensure_knowledge_index(create_if_missing=true)`，再顺序处理本地 Markdown。
11. 用 fake vectorizer/fake transactional Redis 验证 add、unchanged、update、embedding failure、transaction failure 和 retry。
12. 运行：

```powershell
python -m pytest -o addopts="" -q tests/unit/pipelines/ingestion
python -m compileall redis_sre_agent tests
```

Expected: 直接 Markdown 可写入 original schema；重复摄取无重复 chunk；任一失败不会让可检索旧知识消失。

---

### Task 3 / R1b: ArtifactStorage、batch manifest 与 prepare-sources CLI

**Files:**
- Modify: `redis_sre_agent/pipelines/scraper/base.py`
- Create: `redis_sre_agent/pipelines/ingestion/pipeline_workflow_mixin.py`
- Modify: `redis_sre_agent/pipelines/ingestion/_processor_impl.py`
- Modify: `redis_sre_agent/pipelines/ingestion/processor.py`
- Create: `redis_sre_agent/cli/pipeline.py`
- Modify: `redis_sre_agent/cli/main.py`
- Create: `tests/unit/pipelines/ingestion/test_artifact_pipeline.py`
- Create: `tests/unit/cli/test_pipeline_stage8.py`

**Steps:**

1. 写 artifact JSON round-trip、batch manifest、prepare-only 和 ingest-prepared-batch 失败测试。
2. 从 original `scraper/base.py` 补回 `ArtifactStorage`，只保留本地文件与 manifest 能力。
3. 创建裁剪版 `PipelineWorkflowMixin`，只保留 Markdown load、`prepare_source_artifacts()`、`ingest_prepared_batch()` 和 batch list/reindex 必需方法。
4. 严禁复制 original 顶层 `redis_sre_agent.skills.discovery` imports、`_configured_skill_roots()` 或 Agent Skills package expansion。
5. 添加模块导入测试：当前项目不存在 `redis_sre_agent.skills` 时，pipeline/CLI 仍能成功 import。
6. `cli/pipeline.py` 首期只暴露 `prepare-sources` 和 `ingest`；不恢复 scrape/full/status/cleanup/orchestrator。
7. 在 LazyGroup 注册 `pipeline`；未启用 RAG 时命令给出“请先启用”的明确错误，enabled/not-ready 时给出具体安全原因。
8. 所有 CLI 测试使用 fake storage/index/vectorizer，不访问宿主 Redis 或外部 API。
9. 运行：

```powershell
python -m pytest -o addopts="" -q tests/unit/pipelines/ingestion/test_artifact_pipeline.py tests/unit/cli/test_pipeline_stage8.py
python -m compileall redis_sre_agent tests
```

Expected: artifact/batch 能力接回但不存在 skills 隐藏依赖；CLI import 和普通诊断不受影响。

---

### Task 4 / R2: 最小向量检索、Knowledge Provider 与 CLI

**Files:**
- Create: `redis_sre_agent/core/knowledge_helpers.py`
- Modify: `redis_sre_agent/tools/knowledge/knowledge_base.py`
- Create: `redis_sre_agent/cli/knowledge.py`
- Modify: `redis_sre_agent/cli/main.py`
- Create: `tests/unit/core/test_knowledge_helpers.py`
- Replace: `tests/unit/tools/test_dummy_knowledge_provider_stage5.py` with `tests/unit/tools/test_knowledge_provider_stage8.py`
- Create: `tests/unit/cli/test_knowledge_stage8.py`

**Original references:**
- `original-redis-sre-agent-main/redis_sre_agent/core/knowledge_helpers.py::search_knowledge_base_helper`
- `original-redis-sre-agent-main/redis_sre_agent/tools/knowledge/knowledge_base.py`
- `original-redis-sre-agent-main/redis_sre_agent/cli/knowledge.py`

**Steps:**

1. 写失败测试，固定 query embedding、VectorRangeQuery/VectorQuery、version/category filter、limit/offset 和 score 归一化。
2. 从 original 复制 `search_knowledge_base_helper()` 所需的最小 helper；删除 backend override、OTel、HybridQuery、RRF、skills、support tickets、pinned context 和直接 ingest helper。
3. 检索前要求 readiness=ready；disabled/not-ready 分别返回明确错误，不能返回 dummy success/空结果。
4. 保留 Provider 当前 `search(query, limit, offset, version, distance_threshold)` 签名，用真实 helper 替换 dummy 返回。
5. Provider 仍只暴露一个 READ search tool。不要复制 original 的 Agent ingest、fragments、skills 或 tickets 工具。
6. 返回结构加入 `retrieval_kind="knowledge_search"` 和 `retrieval_label="Knowledge search"`，并保留完整来源字段。
7. 区分成功命中、成功无匹配、RAG unavailable 三类结果。
8. 增加 `knowledge search` CLI；输出至少包含 title、source、document hash、chunk index、score 和 content preview。
9. 测试三种 ToolManager 状态：disabled 无 knowledge tool；not-ready 无 knowledge tool；ready 才能通过动态工具名调用真实 provider。
10. 通过 `ToolManager.resolve_tool_call()` 验证链路，禁止绕过 provider。
11. 运行：

```powershell
python -m pytest -o addopts="" -q tests/unit/core/test_knowledge_helpers.py tests/unit/tools/test_knowledge_provider_stage8.py tests/unit/cli/test_knowledge_stage8.py tests/unit/tools/test_manager_stage3.py
```

Expected: standalone CLI 与 ToolManager 使用同一检索实现；不可用 RAG 永远不会出现在 LLM 工具列表。

---

### Task 5 / R3: Chat/Triage 调用与 recommendation worker 证据回传

**Files:**
- Modify: `redis_sre_agent/agent/subgraphs/recommendation_worker.py`
- Modify: `redis_sre_agent/agent/langgraph_agent.py`
- Modify only if required: `redis_sre_agent/agent/chat_agent.py`
- Modify: `redis_sre_agent/agent/helpers.py`
- Modify only if required: `redis_sre_agent/agent/models.py`
- Modify: `tests/unit/agent/test_helpers_stage5.py`
- Modify: `tests/unit/agent/test_chat_agent_stage5.py`
- Modify: `tests/unit/agent/test_langgraph_agent_stage5.py`
- Create: `tests/unit/agent/test_recommendation_worker_rag_evidence.py`
- Create: `tests/integration/test_rag_agent_closed_loop.py`

**Existing Chat path:**

```text
ToolManager → main graph tool node → ResultEnvelope
→ signals_envelopes → extract_citations() → AgentResponse.search_results
```

**Required Triage path:**

```text
recommendation worker AIMessage.tool_calls
→ existing ToolNode/adapters → ToolManager
→ worker ToolMessage
→ build_result_envelope(tool name + args + ToolMessage content)
→ RecState.knowledge_envelopes
→ langgraph_agent gathers worker states
→ merge into top-level signals_envelopes
→ AgentResponse.search_results
```

**Steps:**

1. 写 Chat 确定性 fake-LLM E2E，证明 main graph knowledge search 已正常进入顶层 envelope/citation。
2. 写当前必失败的 Triage 测试：worker 查到知识并用于 Recommendation，但未合并前顶层 `search_results` 为空。
3. 给 `RecState` 增加 `knowledge_envelopes: List[Dict[str, Any]]`。
4. 让 `build_recommendation_worker()` 接收 knowledge ToolDefinition 映射。保留当前 ToolNode/adapters 和 ToolManager 调用，不重复执行工具。
5. worker `tools_node` 根据最后一个 AIMessage 的 tool call ID/name/args 匹配 ToolMessage；仅对 knowledge tool 使用现有 `build_result_envelope()`，`expand_evidence` 不算 knowledge citation。
6. 多轮 worker 调用累积 `knowledge_envelopes`，工具失败也保留 error envelope，但 `extract_citations()` 只能从真实 results 派生来源。
7. `langgraph_agent.reasoning_node` 在 `asyncio.gather()` 后收集所有 worker `knowledge_envelopes`，合并到当前 `envelopes` 并返回为 `signals_envelopes`。
8. worker 检索成功后即使 recommendation composer 随后失败进入 deterministic fallback，已经执行的 knowledge envelopes 仍必须保留。
9. 不改变 Chat 的 `agent → tools → agent → END` 或 Triage 的 `agent → tools → agent → reasoning → END`。
10. 引用权威来源仍是顶层 `tool_envelopes`；`search_results` 继续由 `extract_citations()` 派生，不新建 citation 数据库。
11. 每条 citation 至少断言：`title`、`source`、`document_hash`、`chunk_index`、`score`、`retrieval_kind`。
12. 验证 RAG disabled 时 Chat/Triage 都看不到 knowledge tool；无实际检索时 `search_results == []`。
13. 用 `CliRunner` 验证 query JSON 同时输出 `response`、`search_results`、`tool_envelopes`、`thread_id`。
14. 运行：

```powershell
python -m pytest -o addopts="" -q tests/unit/agent/test_helpers_stage5.py tests/unit/agent/test_chat_agent_stage5.py tests/unit/agent/test_langgraph_agent_stage5.py tests/unit/agent/test_recommendation_worker_rag_evidence.py tests/integration/test_rag_agent_closed_loop.py
```

Expected: Chat 与 Triage 的每次实际知识检索都进入顶层 evidence/citation 链，且全部工具调用仍经过 ToolManager。

---

### Task 6: 真实运行验证、全量回归与交接文档

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `docs/codex/ROADMAP.md`
- Modify: `docs/codex/CURRENT_STAGE.md`
- Modify: `docs/codex/STAGE_HISTORY.md`

**Steps:**

1. 实施前先检查当前 Redis 的 `MODULE LIST`/`COMMAND INFO FT.SEARCH`。若没有 Search module，不修改普通 Redis fallback。
2. Docker/Redis Stack 可用时在隔离环境运行 R0 integration：FT.CREATE、写固定向量、VectorQuery、来源/score；不可用时明确记录未验证项。
3. 对真实 embedding provider 只做显式 opt-in smoke；没有独立 embedding key 时跳过，不得复用或打印 DeepSeek chat key。
4. README 用中文记录：RAG 三态、embedding/chat 配置分离、本地 Markdown direct ingest、prepare/ingest、knowledge search、Agent query 和结构化引用字段。
5. 更新阶段文档，明确 Stage 8 已完成范围和未完成范围；复杂检索/eval/knowledge management 继续保留插槽。
6. 运行完整验证：

```powershell
python -m pip install -e .
python -m compileall redis_sre_agent tests
python -m pytest -q
git diff --check
git status --short
```

7. 若 Redis Stack 可用，再运行：

```powershell
$env:RUN_RAG_REDIS_INTEGRATION = "1"
python -m pytest -o addopts="" -q -m integration tests/integration/test_rag_redis_stack.py
```

8. 最终 smoke 必须覆盖：

```text
rag_enabled=false → 普通 Redis 诊断、无 knowledge tool

rag_enabled=true + invalid config → not-ready、无 knowledge tool、明确安全错误

rag_enabled=true + ready
→ direct local Markdown ingest
→ optional artifact prepare/ingest
→ knowledge search
→ ChatAgent knowledge tool call
→ Triage recommendation worker knowledge tool call
→ worker evidence 合并到顶层 tool_envelopes
→ AgentResponse.search_results 引用输出
```

9. 最终报告必须列出：改动文件、从 original 复制/适配的部分、仍为插槽的能力、所有验证命令、真实 Redis Stack/embedding 是否实际验证、已知差异。

## 停止条件

- 任一 Task 失败后最多进行 2 次基于错误日志的修复。
- 两次仍失败，停止并保留现场，不得降低断言、绕过 ToolManager、伪造检索结果或继续实现后续增强。
- 若发现必须引入计划外大型依赖、改变 Agent 主图、修改 original、泄露凭据或恢复 Stage 9 平台能力，立即停止并请求用户确认。

## 完成定义

- RAG disabled/not-ready/ready 三态均有确定性测试。
- disabled/not-ready 时 LLM 工具列表中不存在 knowledge search。
- embedding factory 不读取 chat 的 OpenAI/DeepSeek key/base URL。
- knowledge index 创建责任唯一且显式，Redis Stack integration 覆盖 create/write/query。
- 本地 Markdown 可原子、幂等写入 original 兼容 schema；失败不删除可检索旧版本。
- 摄取模块不存在 skills 隐藏 import。
- 检索通过 RedisVL/Redis Vector Search，而不是 SCAN fallback。
- Chat 和 Triage 实际检索都进入顶层 `tool_envelopes/search_results`。
- 现有 target discovery、Redis diagnostics、Thread、DeepSeek/fake LLM 行为无回归。
- 默认测试不联网、不调用真实模型、不泄露任何敏感信息。
