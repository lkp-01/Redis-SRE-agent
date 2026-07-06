"""阶段二资源层包。

资源层位于业务流程的最底部。它负责配置、密钥、Redis key、Redis 客户端、索引插槽、
实例模型和集群模型。上层 Agent 或工具以后只需要调用这里的函数，不需要自己拼 Redis
key，也不需要自己处理敏感字段。

当前阶段故意不引入 ToolManager、Redis 诊断工具、Agent、RAG、MCP 或调度逻辑。
"""

__all__: list[str] = []
