# 当前阶段：Stage 5

## 结论

Stage 5 当前目标是“持久任务执行与可验证入口”的诊断切片收口。主链已经从 CLI 进入
真实 LangGraph/LangChain message runtime，并通过 ToolManager 收集 Redis evidence。配置
DeepSeek key 后，Agent 使用 `deepseek-v4-pro` 主模型和 `deepseek-v4-flash` 副模型；无 key
时继续使用 fake LLM 支撑离线验证。

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
- router 在配置 key 时使用 `deepseek-v4-flash`，调用失败时使用本地 fallback。
- `agent/_compat.py` 只保留 fake tool-calling LLM，不模拟 StateGraph 或消息对象。
- `terminal_synthesis` 的确定性报告只作为 fallback/test helper。
- knowledge、MCP、support package、worker、scheduler、API、RAG 和 evaluation 是后续插槽。

## DeepSeek 运行边界

- `core/llm_helpers.py` 沿用 original 的 main/mini/nano 工厂名称。
- main 使用 `deepseek-v4-pro`；mini/nano 和 main failover 使用 `deepseek-v4-flash`。
- failover 只重试失败的模型调用，不重启 StateGraph，不主动重放已经完成的工具。
- 第一版关闭 thinking mode；`reasoning_content` 跨工具轮次回传尚未恢复。
- 真实 API 测试需要 key 和 `RUN_DEEPSEEK_LIVE_TESTS=1`，默认完整测试不会联网。

## 验证方式

```powershell
python -m compileall redis_sre_agent tests
python -m pytest -q
redis-sre-agent query "为什么 Redis 慢"
```

默认测试使用 fake Redis client、fake LLM 或 fake backend；显式标记的 DeepSeek integration
测试可以联网，但不得打印 key、请求头或完整模型响应。
