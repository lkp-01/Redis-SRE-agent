# 当前阶段：Stage 5

## 结论

Stage 5 当前目标是“持久任务执行与可验证入口”的诊断切片收口。主链已经从 CLI 进入
真实 LangGraph/LangChain message runtime，并通过 ToolManager 收集 Redis evidence。

## 主链路

```text
Click LazyGroup
-> redis_sre_agent.cli.query
-> ThreadManager.create_thread 或 get_thread
-> append user message
-> route_to_appropriate_agent
-> ChatAgent 或 SRELangGraphAgent
-> StateGraph agent node / tool node / conditional edge
-> ToolManager
-> always-on TargetDiscoveryToolProvider
-> resolve_redis_targets
-> bind active target
-> RedisCommandToolProvider
-> info / memory_stats / client_list / slowlog
-> ResultEnvelope / ToolMessage evidence
-> AgentResponse
-> assistant message / decision trace
-> CLI JSON output
```

## 保留的裁剪

- Thread/Task 当前是轻量内存实现，接口形状贴近 original，完整 Redis 持久化后续恢复。
- router 在没有真实 LLM 注入时使用本地 fallback，测试不触发 OpenAI API。
- `agent/_compat.py` 只保留 fake tool-calling LLM，不模拟 StateGraph 或消息对象。
- `terminal_synthesis` 的确定性报告只作为 fallback/test helper。
- knowledge、MCP、support package、worker、scheduler、API、RAG 和 evaluation 是后续插槽。

## 验证方式

```powershell
python -m compileall redis_sre_agent tests
python -m pytest -q
redis-sre-agent query "为什么 Redis 慢"
```

测试必须使用 fake Redis client、fake LLM 或 fake backend，不依赖真实 Redis、OpenAI 或外部网络。
