# Redis SRE Agent 诊断切片

这是 `original-redis-sre-agent-main` 的裁剪复刻项目。当前目标不是复制完整平台，
而是保留 Redis 诊断主链路的架构形状，让本地 CLI 可以稳定走通：

```text
Click LazyGroup
-> cli/query.py
-> Thread
-> router
-> ChatAgent 或 SRELangGraphAgent
-> StateGraph agent/tool loop
-> ToolManager
-> target discovery / RedisCommandToolProvider
-> ResultEnvelope evidence
-> assistant response
-> Thread trace
```

## 当前阶段

Stage 5 已落实“持久任务执行与可验证入口”的裁剪版主链：

- `redis-sre-agent query` 由 Click LazyGroup 延迟加载。
- `cli/query.py` 可以创建或恢复 Thread，保存 user/assistant message，并把工具 evidence 保存为 message trace。
- router 保留 original 的 `AgentType` 形状；测试和无真实 LLM 环境使用 fallback，不访问 OpenAI API。
- `ChatAgent` 和 `SRELangGraphAgent` 都使用真实 `langgraph.graph.StateGraph`，包含 agent node、tool node、conditional edge 和工具循环。
- `ToolManager` 统一加载 always-on target discovery、dummy knowledge slot，以及显式或动态绑定的 Redis command provider。
- no-instance 查询会先调用 `resolve_redis_targets`，解析成功后绑定 active target，再继续调用 `info`、`memory_stats`、`client_list`、`slowlog` 等只读诊断工具。
- explicit instance 查询会直接加载 `RedisCommandToolProvider`，不绕过 ToolManager。
- Redis 工具错误、慢日志命令、CONFIG 敏感项和 manager 兜底错误会做脱敏处理。
- `core/tasks.py` 保留 original 风格 TaskManager 和顶层 task helper，但当前 CLI 同步执行，不启动 worker。

## 与 original 的差异

当前仍是诊断切片，不包含完整生产平台：

- Thread、Task 和 target handle 以轻量内存/fake 后端支撑测试；完整 Redis 持久化和搜索索引是后续阶段。
- MCP、RAG ingestion、support package 解压分析、API、worker、scheduler、OpenTelemetry、evaluation suite 和 UI 都只保留插槽或说明。
- knowledge provider 是 dummy slot，只返回空结果，避免提前实现 RAG。
- terminal synthesis 的确定性报告只用于 fake LLM fallback/test helper；正式主链仍是 StateGraph + ToolManager 工具循环。

## 运行测试

请在 `D:\developer\redis_sre_prac\my_sre_agent` 执行：

```powershell
python -m pip install -e .
python -m compileall redis_sre_agent tests
python -m pytest -q
```

测试全部使用 fake Redis client、fake LLM 或 fake backend，不依赖真实 OpenAI API、真实 Redis 服务或外部网络。

## CLI 示例

查看命令：

```powershell
redis-sre-agent --help
redis-sre-agent query --help
redis-sre-agent version
```

运行一次本地诊断链路：

```powershell
redis-sre-agent query "为什么 Redis 慢"
```

如果已经有实例资源，也可以显式指定：

```powershell
redis-sre-agent query "检查 memory 和 slowlog" --instance-id inst-local-cache --agent chat
```

不要把真实 API key、Redis 密码、token、DSN 或带密码的连接串写进文档、日志、测试输出或提交记录。
