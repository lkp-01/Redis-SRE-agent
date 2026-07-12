# Redis SRE Agent 诊断切片 Roadmap

本复刻项目按阶段恢复 original 的 Redis 诊断主链。每一阶段只恢复当前验收需要的
最小架构，不提前实现后续平台能力。

## 已完成阶段

1. Stage 1：Python 包骨架、CLI 入口、基础导入和编译验证。
2. Stage 2：配置、密钥、Redis 资源层、实例/集群模型和安全查询 helper。
3. Stage 3：target discovery/binding、ToolManager 路由、AgentResponse 边界。
4. Stage 4：RedisCommandToolProvider 只读诊断工具。
5. Stage 5：CLI query、Redis Thread、router、真实 StateGraph、target discovery 与
   Redis provider 的可验证诊断主链。
6. Stage 8：显式可选的最小 RAG 闭环：本地 Markdown/artifact 摄取、独立 embedding
   配置、RedisVL 向量索引、knowledge search、Chat/Triage 引用回传。

## 当前 Stage 8 范围

Stage 8 没有替换 Stage 5，而是在其旁边增加受 readiness 门控的 knowledge 路径：

```text
RAG disabled
-> ToolManager 只加载 target discovery / Redis diagnostics

RAG ready
-> local Markdown -> chunk/hash/embed -> atomic Redis Hash replacement
-> sre_knowledge Vector index
-> KnowledgeBaseToolProvider.search -> ToolManager
-> Chat main graph 或 Triage recommendation worker
-> top-level ResultEnvelope -> AgentResponse.search_results
```

关键边界：

- `rag_enabled=false` 时不检查 embedding/Redis Search/index，也不暴露 knowledge tool。
- enabled 但 not-ready 时仍可运行普通 Redis 诊断，但 LLM 看不到 knowledge tool。
- `get_knowledge_index()` 只构造对象；只有显式摄取入口可以创建 index。
- embedding 配置不读取或回退到 DeepSeek chat key/base URL。
- knowledge 读路径只使用 Redis Vector Search，不复用普通目录的 SCAN fallback。
- Recommendation worker evidence 必须合并回顶层 `signals_envelopes`。

## 后续插槽

- HybridQuery、RRF、reranker、完整全文/精确短语检索。
- 独立 Knowledge Agent、Agent 可调用的 ingest、knowledge management/pack。
- skills、support tickets、网页/PDF/Notebook/Cloud ingestion。
- MCP 完整生态、support package 解压和离线分析。
- API、worker、scheduler、OpenTelemetry、完整 evaluation suite、UI。

这些能力继续保留 original 风格的边界，不在 Stage 8 提前实现。
