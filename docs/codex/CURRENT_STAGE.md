# 当前阶段：Stage 8

## 结论

Stage 8 已在现有 Stage 5 诊断主链上接入“本地 Markdown → embedding → Redis Vector
Search → ToolManager knowledge search → Agent 顶层引用”的最小闭环。RAG 默认关闭，
普通诊断不因此增加 Redis Search 或 embedding 依赖。

## RAG 状态与索引责任

- `disabled`：不做 readiness 检查，不加载 knowledge provider。
- `not_ready`：配置、Search/Vector、index 或 schema 不满足契约；不暴露 tool，并返回
  `embedding_config_invalid`、`redis_search_unavailable`、`index_missing` 或
  `schema_mismatch` 等脱敏 reason code。
- `ready`：只有此状态才注册唯一只读 `knowledge_*_search`。

`get_knowledge_index()` 不创建索引。普通 status/Agent/knowledge search 都使用只读检查；
direct/prepared pipeline 摄取是唯一调用 `ensure_knowledge_index(create_if_missing=true)`
的路径。

## 摄取与检索主链

```text
source_documents/**/*.md
-> stable relative source identity
-> DocumentProcessor (1000 chars / 200 overlap)
-> all chunk hashes + all embeddings + dimension validation
-> one Redis MULTI/EXEC
   delete old chunks + HSET new chunks + metadata + source tracking
-> RedisVL VectorRangeQuery / VectorQuery
-> KnowledgeBaseToolProvider.search
-> ToolManager
```

摄取失败时，embedding 阶段没有 Redis mutation；事务未提交时旧版本保持可检索；
网络结果未知时相同输入可用确定性 key 重试。检索没有 SCAN、HybridQuery 或 RRF fallback。

## Agent evidence 链

- Chat：主图 knowledge ToolMessage 直接转换为顶层 ResultEnvelope。
- Triage：recommendation worker 将实际 knowledge ToolMessage 转为
  `RecState.knowledge_envelopes`，worker 完成后立即合并回顶层 `signals_envelopes`。
- composer 失败不会移除已合并 evidence。
- `AgentResponse.search_results` 无条件从顶层成功 knowledge envelopes 重新派生；调用方
  传入的独立 `search_results` 不被信任。

## 当前真实运行边界

2026-07-12 本机检查结果：

- 宿主 Redis 可连接，但 `MODULE LIST` 没有 Search，`FT._LIST` 不可用。
- Docker CLI 已安装，但 Docker service/daemon 未运行；没有本地 Redis Stack/Podman binary。
- 没有独立 embedding key，chat key 未被复用。
- 因此真实 `FT.CREATE`/Hash write/VectorQuery 和真实 embedding smoke 本轮没有执行；
  对应 opt-in integration test 保留为 skip，而不是报告成功。

## 有意保留的裁剪

- 不新增 Knowledge Agent，不把 ingest 暴露给 LLM。
- 不恢复 skills、support tickets、knowledge pack、网页抓取和多索引 pipeline。
- 不恢复 Hybrid/RRF/reranker、API、worker、scheduler、完整 eval 或 UI。
- Chat/Triage 主 StateGraph、ToolManager 路由、target binding、Redis diagnostics 与
  Redis Thread 生命周期保持原结构。
