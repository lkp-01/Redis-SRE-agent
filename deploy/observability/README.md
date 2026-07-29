# 本地可观测性环境

这个目录提供诊断切片的本地测试依赖：三个 Redis 节点、Redis exporter、Prometheus、Loki 和 Grafana Alloy。它们是手动启动的开发环境，不属于 Agent 运行时，也不会由 Agent 自动管理。

## 启动

先确保外部网络存在，再启动服务：

```powershell
docker network inspect redis-sre-lab-net *> $null
if ($LASTEXITCODE -ne 0) { docker network create redis-sre-lab-net }
docker compose -f deploy/observability/compose.yaml up -d
```

本机入口：

- Prometheus：`http://127.0.0.1:19090`
- Loki：`http://127.0.0.1:3100`
- Alloy 管理页面：`http://127.0.0.1:12345`

Loki 和 Alloy 均没有配置认证，只适合本机开发。端口只绑定到 `127.0.0.1`。Alloy 为读取容器日志挂载了 Docker socket；即使使用只读挂载，它仍是高权限接口，不应照搬到共享或生产环境。

## 验证

```powershell
Invoke-RestMethod http://127.0.0.1:3100/ready
Invoke-RestMethod http://127.0.0.1:3100/loki/api/v1/labels

$query = [uri]::EscapeDataString('{job="redis"}')
Invoke-RestMethod "http://127.0.0.1:3100/loki/api/v1/query_range?query=$query&limit=20"
```

Alloy 只采集 Compose 项目 `redis-sre-observability` 中的 `redis-primary`、`redis-replica1` 和 `redis-replica2`。写入 Loki 的稳定标签包括 `job="redis"`、`service`、`instance` 和 `redis_role`。

如果宿主机配置了 `HTTP_PROXY` 或 `HTTPS_PROXY`，运行 Agent 或 Loki 集成测试前还需让
回环地址绕过代理：

```powershell
$env:NO_PROXY = "127.0.0.1,localhost"
```

## 停止

```powershell
docker compose -f deploy/observability/compose.yaml down
```

普通 `down` 会保留命名 volume 中的数据。本环境不包含 Host Telemetry、Support Package、认证扩展、Grafana dashboard 或告警规则。
