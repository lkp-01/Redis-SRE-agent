# Stage History

## 2026-07-12 Live smoke compatibility fixes

### 完成内容

- 为现有 `LightweightSearchIndex.query()` 增加小型 Hash 目录的 `SCAN + HGETALL` fallback，支持当前资源层使用的 CountQuery、FilterQuery、AND TAG 过滤、排序和分页。
- 当前系统 Redis 没有 RediSearch module，fallback 不新增 redisvl 依赖，也不恢复完整全文/向量搜索。
- DeepSeek 的 TopicsList 与 Recommendation structured output 显式改用 `method="function_calling"`，避免不支持的 response_format/json_schema。
- CLI JSON 输出根据 stdout 编码使用 `backslashreplace`；Windows GBK 下中文仍可输出，emoji 等不可编码字符转为合法 JSON escape。

### 对抗性审查结论

- 没有增加 Repository、ORM、新索引层或依赖。
- 资源调用方仍保持 original 的 `index.query()` 形状；Triage 仍保持 original structured-output 主线。
- fallback 使用增量 SCAN，不使用阻塞式 KEYS；只扫描 schema prefix。
- LLM Provider、模型、工具循环和 failover 均未改变。

### 验证结果

- focused Redis/structured/GBK tests：通过。
- target discovery live：自然语言匹配 1 个实例，生成 binding 和 generation，随后动态加载 Redis provider 并执行 `INFO memory`。
- DeepSeek live structured：TopicsList 与 Recommendation 均返回对应 Pydantic model；完整 Triage 不再出现 HTTP 400。
- Thread live：第二个独立进程恢复 4 条历史消息、1 个 target binding、target handle 和 generation=1。
- `python -m compileall redis_sre_agent tests`：通过。
- `git diff --check`：通过。
- 全量测试：88 passed，2 skipped；跳过项为既有显式 live DeepSeek 测试。

### 已知限制

- SCAN fallback 是 O(N)，适合当前裁剪版的小型实例/集群目录，不替代完整 RediSearch。
- Recommendation live smoke 仍观察到无知识来源时生成未被 evidence/citation 支持的写命令建议；命令没有执行，但后续需要在 recommendation worker 内加强 grounding，不应通过引入 Safety Corrector 绕开问题。

## 2026-07-12 LLM 最终回答与 Redis Thread 持久化

### 完成内容

- Chat 保持 `agent -> tools -> agent -> END`；正常结束直接返回工具循环后的最后一条 AIMessage。
- Chat 达到最大迭代且没有终态文本时，使用 messages 与 tool envelopes 做 LLM terminal synthesis；只有异常或空文本才使用确定性报告。
- Triage 保持 `agent -> tools -> agent -> reasoning -> END`；reasoning 恢复 evidence summary、TopicsList、severity 排序/限量、Recommendation worker 和统一 Markdown composer。
- TopicsList 为空或提取失败时先做 LLM terminal synthesis；composer/terminal synthesis 异常时才进入确定性降级。
- 新增裁剪版 `agent/subgraphs/recommendation_worker.py`，保留 original 的短工具循环和 structured Recommendation；只使用现有 evidence、现有 knowledge 插槽与本地 `expand_evidence`。
- 删除 Thread 的 `_THREADS`、`_MESSAGE_TRACES` 进程内 backend，改用系统 Redis List/Hash/String。
- Thread 与 message 使用 26 位 ULID 形状；Thread TTL 为 24 小时，message trace TTL 为 7 天。
- CLI 使用系统 Redis client；从 Redis 恢复 session/user/instance/cluster/context，只回灌 user/assistant 历史；Agent 成功后一次追加本轮 user/assistant，trace 按 assistant message ID 独立保存。
- 两次独立 Python CLI 进程已验证历史、context、instance、target bindings、toolset generation 与 message trace 均可跨进程恢复。

### 从 original 复制或适配的函数形状

- `agent/chat_agent.py`：`_reached_iteration_limit()`、`_synthesize_iteration_limit_response()`、`process_query()` 的终态提取顺序。
- `agent/langgraph_agent.py`：`_summarize_envelopes_for_reasoning()`、`_build_expand_evidence_tool()`、`_compose_final_markdown()` 与 `reasoning_node()` 的 topic map/reduce 主线。
- `agent/subgraphs/recommendation_worker.py`：`RecState`、`build_recommendation_worker()`、`llm_node()`、`tools_node()`、`should_continue()`、`synth_node()`。
- `agent/terminal_synthesis.py`：`TerminalSynthesisConfig`、消息/evidence 格式化和 `synthesize_terminal_response()`。
- `core/threads.py`：`Message`、`ThreadMetadata`、`Thread` 与 ThreadManager 的 create/get/update/append/save/trace 方法。
- `core/keys.py`：`message_decision_trace()` 与 original Thread key 字符串。
- `core/redis.py`：`SRE_THREADS_SCHEMA`、`get_threads_index()`。
- `cli/query.py`：系统 Redis client、Thread 恢复、history 转换、Agent 调用、成对消息追加和 trace 保存顺序。

### 最小兼容适配

- 当前依赖没有 `ulid`，且本阶段禁止修改依赖配置，因此使用标准库生成相同的 26 位 Crockford Base32 ULID 形状。
- 当前没有完整知识 Provider；Recommendation worker 不虚构 Provider，只保留现有 knowledge 插槽和本地 `expand_evidence`。
- Thread subject 保留确定性截断，不恢复与本阶段无关的 LLM 标题生成。

### 有意保留的插槽

- LangGraph Redis checkpoint/resume、API、Docket worker、scheduler。
- MCP、RAG ingestion、Knowledge Agent、Safety Fact Corrector、Agent Memory。
- Feedback、Observability、Support Package。
- 完整 RedisVL Thread 搜索/list/delete 行为；本阶段只补 schema 与 index 入口。

### 验证结果

- Agent 新增终态/structured pipeline 测试：通过。
- Redis Thread List/Hash/String、TTL、ULID、跨 manager 测试：通过。
- 两次独立 CLI 进程真实系统 Redis 测试：通过。
- Stage 5 target discovery、ToolManager、RedisCommandToolProvider 回归：通过。
- `python -m compileall redis_sre_agent tests`：通过。
- `git diff --check`：通过。
- 全量测试：88 collected，86 passed，2 skipped；跳过项为既有显式 live DeepSeek 测试。

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
