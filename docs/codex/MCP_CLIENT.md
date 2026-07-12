# Stage 9：外部 MCP Client 只读切片

## 目标和调用链

Stage 9 让 Agent 以 MCP Client 身份连接进程启动时受信任配置中的外部 MCP Server，
但不改变 Redis 诊断主链。所有 MCP 工具都先注册到当前轮的 `ToolManager`，执行时仍走：

```text
LLM tool call
-> ToolManager.resolve_tool_call()
-> MCPToolProvider.call_tool()
-> MCP session
-> ToolMessage
-> ResultEnvelope
```

MCP 未配置、配置无效或 server 不可用时，只跳过对应 provider；target discovery、动态
Redis provider、可选 knowledge provider 和已有诊断仍可继续工作。

## stdio 配置示例

以下内容可以放在 `config.yaml`，再通过 `SRE_AGENT_CONFIG` 指向该文件。命令与入口文件
必须来自受信任的部署配置，不能由用户 query、LLM 参数或 Thread context 覆盖。

```yaml
mcp_servers:
  local_observer:
    command: python
    args:
      - path/to/trusted_readonly_server.py
    env:
      MCP_OBSERVER_TOKEN: "${MCP_OBSERVER_TOKEN}"
    tools:
      read_status:
        capability: diagnostics
        description: "读取外部系统的当前状态。"
        action_kind: read
```

stdio 子进程不会继承完整父进程环境。客户端只保留启动所需的 `PATH`，Windows 下再保留
`SYSTEMROOT`、`COMSPEC`、`TEMP`、`TMP`，并加入当前 server 显式配置的 `env`。
`${ENV_VAR}` 在连接前展开；变量缺失时该 provider 以脱敏错误码失败，不启动子进程。

## Streamable HTTP 配置示例

```yaml
mcp_servers:
  remote_observer:
    url: https://mcp.example.invalid/mcp
    transport: streamable_http
    headers:
      Authorization: "Bearer ${MCP_ACCESS_TOKEN}"
    tools:
      read_incident:
        capability: diagnostics
        description: "读取外部事件的当前状态。"
        action_kind: read
```

URL transport 只接受 `sse` 或 `streamable_http`，省略时使用
`streamable_http`。默认要求 HTTPS；`http://localhost`、loopback IPv4/IPv6 可用于本地
开发。其他明文 HTTP 只有在受控网络中显式设置下列开关才会通过校验：

```yaml
mcp_servers:
  controlled_lab:
    url: http://mcp.lab.example.invalid/mcp
    transport: streamable_http
    allow_insecure_http: true
    tools:
      read_status:
        action_kind: read
```

该开关会降低传输安全性，不应作为生产默认值。

## 必需的只读 allowlist

- `tools` 缺失或为空时，一个远端工具也不会暴露。
- 远端工具名必须与 `tools` 中的 key 精确匹配。
- 只有显式 `action_kind: read` 才能注册；`write`、`unknown` 和未填写值均被拒绝。
- `capability` 和 `description` 可以覆盖远端目录，但不能把非 READ 工具变成隐式可用。
- 工具名会生成确定性的安全 slug；同一 server 的重复原始名、规范名冲突，以及与内建
  target/Redis/knowledge/其他 MCP 工具冲突时，整个 provider fail-closed，不覆盖旧工具。
- LLM 工具超过 64 个时，target discovery、Redis diagnostics、knowledge 和其他内建
  provider 优先，MCP 工具最后裁剪。

## 生命周期和边界

每个 `ToolManager` 使用自己的 `AsyncExitStack` 管理 MCP transport、`ClientSession`、
stream 和 stdio 子进程。当前轮结束时统一关闭；下一轮重新连接并发现工具。当前实现没有
进程级连接池、后台健康检查或跨会话永久 MCP cache。

远端 description 最多 4000 字符，输入 schema 最多 64 KiB，structured/text 结果合计
最多 32000 字符。图片、音频、二进制 resource 只返回类型和数量，不把 base64 或 resource
正文送入 LLM。连接、发现、调用和关闭均受 `tool_timeout` 限制；稳定错误码包括：

- `mcp_connect_failed`
- `mcp_discovery_failed`
- `mcp_timeout`
- `mcp_tool_error`
- `mcp_invalid_response`

日志和返回错误不会包含 command、URL、args、headers、env 值、工具调用参数或远端异常原文。

## 当前明确不支持

- MCP WRITE/UNKNOWN 工具、人工审批和中断恢复。
- 全局 `MCPConnectionPool`、后台重连和动态 catalog refresh。
- OAuth、token refresh 和交互式认证。
- Agent 作为 MCP Server；没有恢复 MCP server CLI/API/worker。
- `langchain-mcp-adapters` 或第二套路由抽象。
- 图片、音频、二进制 resource 内容注入 LLM。

默认测试只使用 transport mock 与本地 fake stdio server，不访问公网 MCP Server、真实模型
或其他外部 API。SSE/Streamable HTTP 已做 mock 协议分支验证，但本阶段没有执行公网 smoke。
