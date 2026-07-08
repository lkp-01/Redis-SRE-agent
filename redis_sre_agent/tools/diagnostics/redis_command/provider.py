"""Redis command diagnostics provider 的阶段三裁剪版。

文件、类名、`__init__` 签名和诊断方法名沿用原项目。阶段三只暴露原名 `info`
作为 dummy 工具，用来证明 target-scoped provider 能被 ToolManager 加载和调用；
其余真实 Redis 在线诊断命令保留同名轻量插槽，避免提前进入 Stage 4。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from redis.asyncio import Redis

from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.tools.decorators import status_update
from redis_sre_agent.tools.models import SystemHost, ToolCapability, ToolDefinition
from redis_sre_agent.tools.protocols import ToolProvider


class RedisCliConfig(BaseModel):
    """Redis command provider 配置插槽。"""

    pass


class RedisCommandToolProvider(ToolProvider):
    """Redis command diagnostics provider 的阶段三占位实现。"""

    capabilities = {ToolCapability.DIAGNOSTICS}

    def __init__(
        self,
        redis_instance: Optional[RedisInstance] = None,
        connection_url: Optional[str] = None,
        _config: Optional[RedisCliConfig] = None,
    ):
        super().__init__(redis_instance)
        self.connection_url = "[stage3-redacted]" if (redis_instance or connection_url) else ""
        self._client = None

    @property
    def provider_name(self) -> str:
        return "redis_command"

    @property
    def requires_redis_instance(self) -> bool:
        return True

    def get_client(self) -> Redis:
        """真实 Redis client 懒加载入口，Stage 4 再补回。"""
        raise NotImplementedError("Redis client access is a Stage 4 slot.")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self._client = None

    def create_tool_schemas(self) -> List[ToolDefinition]:
        """阶段三只暴露原名 INFO 工具。"""
        return [
            ToolDefinition(
                name=self._make_tool_name("info"),
                description=(
                    "Execute Redis INFO command to get server statistics and information. "
                    "Stage three returns a deterministic dummy report."
                ),
                capability=ToolCapability.DIAGNOSTICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "description": "Optional INFO section to query.",
                        }
                    },
                    "required": [],
                },
            )
        ]

    def get_status_update(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        operation = tool_name.split("_")[-1]
        if operation == "info":
            section = args.get("section")
            if section:
                return f"I'm running Redis INFO for the {section} section."
            return "I'm running Redis INFO to collect server metrics."
        if operation == "slowlog":
            return "I'm checking Redis SLOWLOG for slow queries."
        if operation == "client" and "list" in tool_name:
            return "I'm listing connected Redis clients."
        if operation == "get" and "config" in tool_name:
            pattern = args.get("pattern", "*")
            return f"I'm inspecting Redis configuration with CONFIG GET {pattern}."
        return None

    def resolve_operation(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        op = tool_name.split("_")[-1]
        if op == "info":
            if "cluster" in tool_name:
                return "cluster_info"
            if "replication" in tool_name:
                return "replication_info"
            if "index" in tool_name:
                return "search_index_info"
            return "info"
        if op == "slowlog":
            return "slowlog"
        if op == "log" and "acl" in tool_name:
            return "acl_log"
        if op == "get" and "config" in tool_name:
            return "config_get"
        if op == "list" and "client" in tool_name:
            return "client_list"
        if op == "stats" and "memory" in tool_name:
            return "memory_stats"
        if op == "keys":
            return "sample_keys"
        if op == "indexes":
            return "search_indexes"
        return op

    @status_update("I'm running Redis INFO to collect server metrics ({section}).")
    async def info(self, section: Optional[str] = None) -> Dict[str, Any]:
        """原名 INFO 工具的确定性 dummy 实现。"""
        if self.redis_instance is None:
            return {
                "status": "error",
                "tool": "info",
                "message": "No Redis target is bound.",
            }
        return {
            "status": "success",
            "tool": "info",
            "mode": "mock",
            "section": section or "default",
            "target_handle": self.redis_instance.id,
            "target_name": self.redis_instance.name,
            "info": {
                "redis_version": "stage3-dummy",
                "role": "unknown",
                "connected_clients": 0,
            },
        }

    async def slowlog(self, count: int = 10) -> Dict[str, Any]:
        return {"status": "not_implemented", "tool": "slowlog", "stage": 3}

    async def acl_log(self, count: int = 10) -> Dict[str, Any]:
        return {"status": "not_implemented", "tool": "acl_log", "stage": 3}

    async def config_get(self, pattern: str) -> Dict[str, Any]:
        return {"status": "not_implemented", "tool": "config_get", "stage": 3}

    async def client_list(self, client_type: Optional[str] = None) -> Dict[str, Any]:
        return {"status": "not_implemented", "tool": "client_list", "stage": 3}

    async def cluster_info(self) -> Dict[str, Any]:
        return {"status": "not_implemented", "tool": "cluster_info", "stage": 3}

    async def replication_info(self) -> Dict[str, Any]:
        return {"status": "not_implemented", "tool": "replication_info", "stage": 3}

    async def memory_stats(self) -> Dict[str, Any]:
        return {"status": "not_implemented", "tool": "memory_stats", "stage": 3}

    async def sample_keys(self, count: int = 100) -> Dict[str, Any]:
        return {"status": "not_implemented", "tool": "sample_keys", "stage": 3}

    async def search_indexes(self) -> Dict[str, Any]:
        return {"status": "not_implemented", "tool": "search_indexes", "stage": 3}

    async def search_index_info(self, index_name: str) -> Dict[str, Any]:
        return {"status": "not_implemented", "tool": "search_index_info", "stage": 3}

    async def system_hosts(self) -> List[SystemHost]:
        return []
