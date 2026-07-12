# Stage History

## 2026-07-12 Stage 5 DeepSeek LLM integration

### 完成内容

- 增加 `langchain-openai` 依赖，恢复 original 风格的 main/mini/nano LLM 工厂。
- 默认配置 `deepseek-v4-pro` 主模型和 `deepseek-v4-flash` 副模型/轻量模型。
- Pro 单次调用失败后将同一输入切换给 Flash，不重启 StateGraph 或主动重放 Redis 工具。
- ChatAgent、SRELangGraphAgent 和 router 接入真实模型工厂；显式 LLM 注入仍具有最高优先级。
- 无 key 时保留 fake tool-calling LLM；router 模型失败时保留确定性路由 fallback。
- 增加不含密钥的 `.env.example`、单元测试和显式启用的真实 DeepSeek smoke test。
- 第一版显式关闭 thinking mode，避免工具轮次遗漏 `reasoning_content`。

### 从 original 复制或适配的形状

- `core/llm_helpers.py` 的 `create_llm`、`create_mini_llm`、`create_nano_llm` 公共名称。
- `ChatOpenAI(model/api_key/base_url/timeout)` 集中创建边界。
- ChatAgent/SRELangGraphAgent 使用 main，router 使用 nano 的职责划分。

### 有意保留的插槽

- thinking mode 的 `reasoning_content` 跨工具轮次回传。
- 多 base URL、多 key、多供应商主副容灾。
- 指数退避、熔断、健康状态、token/费用/延迟观测。
- streaming、完整 structured output wrapper、DeepSeek strict schema。

### 验证结果

- 改动前基线：67 passed。
- 新增配置/工厂测试：10 passed。
- Agent/router/Stage 5 E2E focused：18 passed。
- 完整测试：78 passed，2 skipped。
- 2 个 live tests 因本机没有 DeepSeek key 且未开启 live 开关而跳过，未伪造真实连通性结论。

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
