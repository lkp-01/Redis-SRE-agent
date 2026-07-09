# Redis SRE Agent 诊断切片 Roadmap

本复刻项目按阶段恢复 original 的 Redis 诊断主链。每一阶段只恢复当前验收需要的
最小架构，不提前实现后续平台能力。

## 已完成阶段

1. Stage 1：Python 包骨架、CLI 入口、基础导入和编译验证。
2. Stage 2：配置、密钥、Redis 资源层、实例/集群模型和安全查询 helper。
3. Stage 3：target discovery/binding、ToolManager 路由、AgentResponse 边界。
4. Stage 4：RedisCommandToolProvider 只读诊断工具，包括 `INFO`、`SLOWLOG`、`CLIENT LIST`、`MEMORY STATS` 等。
5. Stage 5：CLI query 到 Thread、router、真实 StateGraph Agent、ToolManager、target discovery 和 Redis provider 的可验证主链。

## 当前 Stage 5 范围

Stage 5 的核心目标是让这条链路可运行、可测试、可解释：

```text
redis-sre-agent query "为什么 Redis 慢"
-> Click LazyGroup
-> cli/query.py
-> Thread user message
-> router
-> ChatAgent 或 SRELangGraphAgent
-> StateGraph agent/tool loop
-> ToolManager
-> resolve_redis_targets 或 explicit instance provider load
-> RedisCommandToolProvider
-> ResultEnvelope evidence
-> assistant message / trace
```

## 后续插槽

以下能力不属于 Stage 5，不应在当前阶段补完整实现：

- API、worker、scheduler。
- MCP provider 完整加载和外部工具生态。
- RAG ingestion、embedding、vector retrieval。
- support package 解压和离线分析。
- OpenTelemetry、完整 evaluation suite、UI。

这些模块只保留 original 风格的边界、命名和扩展点。
