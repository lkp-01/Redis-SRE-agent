# Stage History

## 2026-07-09 Stage 5 final closure

### 完成内容

- 恢复 `redis-sre-agent query` 的 Click LazyGroup 入口和 `cli/query.py` 生命周期。
- query 可以创建或恢复 Thread，保存 user/assistant message，并用 assistant message id 关联 decision trace。
- ChatAgent 和 SRELangGraphAgent 都使用真实 `StateGraph`，保留 agent node、tool node、conditional edge 和 tool loop。
- ToolManager 统一加载 target discovery、dummy knowledge slot 和 Redis command provider。
- explicit instance 路径直接经 ToolManager 进入 RedisCommandToolProvider。
- no-instance 路径先调用 `resolve_redis_targets`，解析成功后绑定 target，再进入 Redis 诊断工具。
- Redis evidence 覆盖 `resolve_redis_targets`、`info`、`memory_stats`、`client_list`、`slowlog`。
- Redis provider 和 ToolManager 兜底错误会清洗 password、secret、token 和 Redis URL。
- TaskManager 保留 original 风格的 create/get/list/delete 轻量接口，供后续 worker 阶段接回。

### 从 original 复制或适配的形状

- `cli/main.py` 的 LazyGroup 注册和延迟加载结构。
- `cli/query.py` 的 Thread、router、Agent、trace 保存边界。
- `agent/chat_agent.py` 与 `agent/langgraph_agent.py` 的 StateGraph agent/tool loop。
- `agent/tool_execution.py` 的 ToolManager -> ToolMessage 边界。
- `tools/manager.py` 的 provider lifecycle、routing table、dynamic target attachment。
- `core/threads.py`、`core/tasks.py`、`core/targets.py` 的 Thread/Task/Target 公共模型和 helper 命名。

### 有意保留的插槽

- API、worker、scheduler。
- MCP provider 完整实现。
- RAG ingestion、embedding、vector retrieval。
- support package 解压分析。
- OpenTelemetry、evaluation suite、UI。

### 验证命令

```powershell
python -m compileall redis_sre_agent tests
python -m pytest -q
```

### 已知差异

- 当前 Thread/Task/target handle 后端是轻量测试实现，不是 original 的完整 Redis 持久化和 RediSearch 索引。
- fake LLM 只用于无真实 LLM 的本地验证；正式结构仍经过真实 LangGraph runtime。
- knowledge provider 返回空结果，只作为 Stage 5 RAG 插槽。
