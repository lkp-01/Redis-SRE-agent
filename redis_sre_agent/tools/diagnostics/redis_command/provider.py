"""Redis command diagnostics provider。

第四阶段把阶段三的 dummy 工具补回为真实 redis-py 只读命令适配器。这里仍只负责
收集结构化 evidence，不负责 Agent 推理、根因判断或报告生成。
"""

from __future__ import annotations

import heapq
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, SecretStr
from redis.asyncio import Redis

from redis_sre_agent.core.instances import RedisInstance, mask_redis_url
from redis_sre_agent.tools.decorators import status_update
from redis_sre_agent.tools.models import SystemHost, ToolCapability, ToolDefinition
from redis_sre_agent.tools.protocols import ToolProvider

logger = logging.getLogger(__name__)

_COMMAND_UNAVAILABLE_PATTERNS = (
    "not allowed",
    "unknown command",
    "disabled",
    "noperm",
    "no permission",
)
_REDACTED = "[REDACTED]"
_SENSITIVE_CONFIG_KEYS = {"requirepass", "masterauth"}
_SENSITIVE_KEY_MARKERS = ("password", "secret", "token")
_SENSITIVE_COMMANDS = {"auth", "hello"}


def _is_command_unavailable_error(exc: Exception) -> bool:
    """判断 Redis 命令是否因为版本、模块或权限限制不可用。"""
    lowered = str(exc).lower()
    return any(pattern in lowered for pattern in _COMMAND_UNAVAILABLE_PATTERNS)


def _safe_error_message(exc: Exception) -> str:
    """清洗异常文本，避免 Redis URL、密码、token 进入工具返回或日志。"""
    message = str(exc)
    message = re.sub(
        r"(?i)\b(rediss?|unix)://[^\s'\"<>@]+@[^\s'\"<>]+",
        lambda match: mask_redis_url(match.group(0)),
        message,
    )
    message = re.sub(
        r"(?i)\b(password|secret|token|requirepass|masterauth|pass)(\s*[=:]\s*)([^\s,;]+)",
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}",
        message,
    )
    return message


def _redact_config_value(key: str, value: Any) -> Any:
    """对 CONFIG GET 中的敏感配置值做遮蔽，保留配置项名称便于审计。"""
    lowered = str(key).lower()
    if lowered in _SENSITIVE_CONFIG_KEYS or any(
        marker in lowered for marker in _SENSITIVE_KEY_MARKERS
    ):
        return _REDACTED
    return value


def _command_part_to_text(part: Any) -> str:
    if isinstance(part, bytes):
        return part.decode("utf-8", errors="replace")
    return str(part)


def _redact_command_args(command: Any) -> str:
    """清洗慢日志命令参数，避免 AUTH、password、token 等值泄漏。"""
    if isinstance(command, (list, tuple)):
        parts = [_command_part_to_text(part) for part in command]
    else:
        parts = _command_part_to_text(command).split()
    if not parts:
        return ""

    first = parts[0].lower()
    redact_all_args = first in _SENSITIVE_COMMANDS
    redact_next = False
    redacted: list[str] = []
    for index, part in enumerate(parts):
        lowered = part.lower()
        if index > 0 and (redact_all_args or redact_next):
            redacted.append(_REDACTED)
            redact_next = False
            continue
        redacted.append(part)
        if (
            lowered in _SENSITIVE_CONFIG_KEYS
            or any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS)
            or lowered.startswith((">", "<"))
        ):
            redact_next = True
    return " ".join(redacted)


class RedisCliConfig(BaseModel):
    """Redis command provider 配置插槽。"""

    pass


class RedisCommandToolProvider(ToolProvider):
    """用 redis-py 执行 Redis 只读诊断命令的 provider。"""

    capabilities = {ToolCapability.DIAGNOSTICS}

    def __init__(
        self,
        redis_instance: Optional[RedisInstance] = None,
        connection_url: Optional[str] = None,
        _config: Optional[RedisCliConfig] = None,
    ):
        super().__init__(redis_instance)

        if redis_instance:
            conn_url = redis_instance.connection_url
            if isinstance(conn_url, SecretStr):
                self.connection_url = conn_url.get_secret_value()
            elif isinstance(conn_url, str):
                self.connection_url = conn_url
            else:
                raise ValueError("connection_url has unexpected type.")
        elif connection_url:
            self.connection_url = connection_url
        else:
            raise ValueError("Either redis_instance or connection_url must be provided")

        if not self.connection_url or not isinstance(self.connection_url, str):
            raise ValueError("Invalid connection_url.")
        if not self.connection_url.startswith(("redis://", "rediss://", "unix://")):
            raise ValueError(
                f"Invalid Redis URL scheme: {mask_redis_url(self.connection_url)}. "
                "Must start with redis://, rediss://, or unix://"
            )

        self._client: Optional[Redis] = None

    @property
    def provider_name(self) -> str:
        return "redis_command"

    @property
    def requires_redis_instance(self) -> bool:
        return True

    def get_client(self) -> Redis:
        """懒加载 Redis client，导入 provider 时不会建立外部连接。"""
        if self._client is None:
            self._client = Redis.from_url(self.connection_url, decode_responses=True)
            logger.info("Connected to Redis at %s", mask_redis_url(self.connection_url))
        return self._client

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
        self._client = None

    def create_tool_schemas(self) -> List[ToolDefinition]:
        """创建 Redis command 诊断工具 schema。"""
        return [
            ToolDefinition(
                name=self._make_tool_name("info"),
                description=(
                    "Execute Redis INFO command to get server statistics and information. "
                    "Use this to check Redis server status, memory usage, client connections, "
                    "replication status, and performance metrics. For actual client counts, "
                    "prefer the 'clients' section. Can query specific sections or get all "
                    "information."
                ),
                capability=ToolCapability.DIAGNOSTICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "description": (
                                "Optional INFO section to query. Examples: 'server', 'memory', "
                                "'clients', 'stats', 'replication', 'cpu', 'keyspace'. "
                                "Leave empty for all sections."
                            ),
                        },
                    },
                    "required": [],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("slowlog"),
                description=(
                    "Query Redis SLOWLOG to find slow queries. Use this to diagnose "
                    "performance issues by identifying commands that took longer than "
                    "the configured slowlog threshold to execute."
                ),
                capability=ToolCapability.DIAGNOSTICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "count": {
                            "type": "integer",
                            "description": "Number of slowlog entries to retrieve (default: 10)",
                            "default": 10,
                        }
                    },
                    "required": [],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("acl_log"),
                description=(
                    "Query Redis ACL LOG to find authentication and authorization failures. "
                    "Use this to diagnose security issues, permission denials, and "
                    "authentication problems."
                ),
                capability=ToolCapability.DIAGNOSTICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "count": {
                            "type": "integer",
                            "description": "Number of ACL log entries to retrieve (default: 10)",
                            "default": 10,
                        }
                    },
                    "required": [],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("config_get"),
                description=(
                    "Get Redis configuration values using CONFIG GET. Use this to inspect "
                    "Redis configuration settings. Supports pattern matching with wildcards."
                ),
                capability=ToolCapability.DIAGNOSTICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": (
                                "Configuration parameter pattern. Examples: 'maxmemory*', "
                                "'timeout', 'save', '*'. Use '*' to get all config values."
                            ),
                        }
                    },
                    "required": ["pattern"],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("client_list"),
                description=(
                    "List connected Redis clients using CLIENT LIST. Use this to diagnose "
                    "connection issues, identify problematic clients, or check client "
                    "connection details. This is the definitive inventory of current client "
                    "connections."
                ),
                capability=ToolCapability.DIAGNOSTICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "client_type": {
                            "type": "string",
                            "description": (
                                "Optional client type filter. Values: 'normal', 'master', "
                                "'replica', 'pubsub'. Leave empty for all clients."
                            ),
                        }
                    },
                    "required": [],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("cluster_info"),
                description=(
                    "Get Redis cluster information using CLUSTER INFO. Use this to check "
                    "cluster state, slots distribution, and cluster health. Only works "
                    "if Redis is running in cluster mode."
                ),
                capability=ToolCapability.DIAGNOSTICS,
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            ToolDefinition(
                name=self._make_tool_name("replication_info"),
                description=(
                    "Get Redis replication information including role, connected replicas, "
                    "and replication lag. Use this to diagnose replication issues."
                ),
                capability=ToolCapability.DIAGNOSTICS,
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            ToolDefinition(
                name=self._make_tool_name("memory_stats"),
                description=(
                    "Get detailed Redis memory statistics using MEMORY STATS. Use this "
                    "for in-depth memory analysis including allocator stats, fragmentation, "
                    "and memory breakdown by category. Do not use this for client counts: "
                    "fields such as clients.normal and clients.slaves are client-memory "
                    "overhead in bytes, not numbers of connected clients. Use INFO clients "
                    "or CLIENT LIST for counts."
                ),
                capability=ToolCapability.DIAGNOSTICS,
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            ToolDefinition(
                name=self._make_tool_name("bigkey_scan"),
                description=(
                    "Safely scan the current Redis database for large keys using cursor-based "
                    "SCAN and MEMORY USAGE. Returns the largest measured keys and keys above "
                    "a byte threshold. The scan is bounded by key and time budgets and never "
                    "uses KEYS."
                ),
                capability=ToolCapability.DIAGNOSTICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "threshold_bytes": {
                            "type": "integer",
                            "description": "Size in bytes at or above which a key is considered big (default: 1048576).",
                            "default": 1048576,
                            "minimum": 0,
                        },
                        "max_keys": {
                            "type": "integer",
                            "description": "Maximum unique keys to inspect (default: 10000; hard maximum: 100000).",
                            "default": 10000,
                            "minimum": 1,
                            "maximum": 100000,
                        },
                        "scan_count": {
                            "type": "integer",
                            "description": "SCAN COUNT hint per batch (default: 500; hard maximum: 2000).",
                            "default": 500,
                            "minimum": 1,
                            "maximum": 2000,
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Number of largest keys to return (default: 20; hard maximum: 100).",
                            "default": 20,
                            "minimum": 1,
                            "maximum": 100,
                        },
                        "time_limit_ms": {
                            "type": "integer",
                            "description": "Approximate scan time budget in milliseconds (default: 5000; hard maximum: 30000).",
                            "default": 5000,
                            "minimum": 100,
                            "maximum": 30000,
                        },
                    },
                    "required": [],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("sample_keys"),
                description=(
                    "Sample random keys from the Redis keyspace with minimal impact using RANDOMKEY. "
                    "Returns a sample of unique keys with their types. Prefer this for lightweight "
                    "inspections of the data model and type distribution."
                ),
                capability=ToolCapability.DIAGNOSTICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "count": {
                            "type": "integer",
                            "description": "Number of keys to sample (default: 100; upper bound enforced)",
                            "default": 100,
                        }
                    },
                    "required": [],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("search_indexes"),
                description=(
                    "List all Redis Search (RediSearch) indexes using FT._LIST. "
                    "Use this to discover what search indexes exist in the Redis instance."
                ),
                capability=ToolCapability.DIAGNOSTICS,
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            ToolDefinition(
                name=self._make_tool_name("search_index_info"),
                description=(
                    "Get detailed information about a Redis Search index using FT.INFO. "
                    "Returns schema, statistics, and configuration for the specified index. "
                    "Use this to understand index structure, document count, and performance metrics."
                ),
                capability=ToolCapability.DIAGNOSTICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "index_name": {
                            "type": "string",
                            "description": "Name of the search index to inspect",
                        }
                    },
                    "required": ["index_name"],
                },
            ),
        ]

    def get_status_update(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """Return a concise natural-language status for this Redis CLI call."""
        operation = tool_name.split("_")[-1]
        if (
            operation == "info"
            and "index" not in tool_name
            and "cluster" not in tool_name
            and "replication" not in tool_name
        ):
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
        """把动态工具名映射回 provider 上的同名方法。"""
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
        if op == "scan" and "bigkey" in tool_name:
            return "bigkey_scan"
        if op == "keys":
            return "sample_keys"
        if op == "indexes":
            return "search_indexes"
        return op

    @status_update("I'm running Redis INFO to collect server metrics ({section}).")
    async def info(self, section: Optional[str] = None) -> Dict[str, Any]:
        """执行 Redis INFO 命令并返回解析后的结构化结果。"""
        logger.info("Executing INFO%s", f" {section}" if section else "")
        try:
            client = self.get_client()
            if section:
                result = await client.info(section)
            else:
                result = await client.info()

            return {
                "status": "success",
                "section": section or "all",
                "data": result,
            }
        except Exception as exc:
            error = _safe_error_message(exc)
            logger.error("Failed to execute INFO: %s", error)
            return {"status": "error", "error": error}

    @status_update("I'm checking Redis SLOWLOG for slow queries.")
    async def slowlog(self, count: int = 10) -> Dict[str, Any]:
        """查询 SLOWLOG，并把命令参数中的敏感值遮蔽。"""
        logger.info("Querying SLOWLOG (count=%s)", count)
        try:
            client = self.get_client()
            result = await client.slowlog_get(count)

            entries = []
            for entry in result:
                entries.append(
                    {
                        "id": entry["id"],
                        "timestamp": entry["start_time"],
                        "duration_us": entry["duration"],
                        "command": _redact_command_args(entry["command"]),
                        "client_address": entry.get("client_address", "N/A"),
                        "client_name": entry.get("client_name", "N/A"),
                    }
                )

            return {"status": "success", "count": len(entries), "entries": entries}
        except Exception as exc:
            error = _safe_error_message(exc)
            logger.error("Failed to query SLOWLOG: %s", error)
            return {"status": "error", "error": error}

    @status_update("I'm checking Redis ACL LOG for auth and permission issues.")
    async def acl_log(self, count: int = 10) -> Dict[str, Any]:
        """查询 ACL LOG，返回认证和权限失败记录。"""
        logger.info("Querying ACL LOG (count=%s)", count)
        try:
            client = self.get_client()
            result = await client.acl_log(count)

            entries = []
            for entry in result:
                entries.append(
                    {
                        "count": entry.get("count", 0),
                        "reason": entry.get("reason", "unknown"),
                        "context": entry.get("context", "unknown"),
                        "object": entry.get("object", "N/A"),
                        "username": entry.get("username", "N/A"),
                        "age_seconds": entry.get("age-seconds", 0),
                        "client_info": entry.get("client-info", "N/A"),
                    }
                )

            return {"status": "success", "count": len(entries), "entries": entries}
        except Exception as exc:
            error = _safe_error_message(exc)
            logger.error("Failed to query ACL LOG: %s", error)
            return {"status": "error", "error": error}

    @status_update("I'm inspecting Redis configuration with CONFIG GET {pattern}.")
    async def config_get(self, pattern: str) -> Dict[str, Any]:
        """执行 CONFIG GET，并遮蔽密码、secret、token 等配置值。"""
        logger.info("Executing CONFIG GET %s", pattern)
        try:
            client = self.get_client()
            result = await client.config_get(pattern)
            safe_config = {
                key: _redact_config_value(key, value) for key, value in (result or {}).items()
            }

            return {
                "status": "success",
                "pattern": pattern,
                "config": safe_config,
                "count": len(result or {}),
            }
        except Exception as exc:
            error = _safe_error_message(exc)
            logger.error("Failed to execute CONFIG GET: %s", error)
            return {"status": "error", "error": error}

    @status_update("I'm listing connected Redis clients.")
    async def client_list(self, client_type: Optional[str] = None) -> Dict[str, Any]:
        """执行 CLIENT LIST，返回当前连接客户端列表。"""
        logger.info("Executing CLIENT LIST%s", f" TYPE {client_type}" if client_type else "")
        try:
            client = self.get_client()
            if client_type:
                result = await client.client_list(_type=client_type)
            else:
                result = await client.client_list()

            return {
                "status": "success",
                "client_type": client_type or "all",
                "count": len(result),
                "clients": result,
            }
        except Exception as exc:
            error = _safe_error_message(exc)
            logger.error("Failed to execute CLIENT LIST: %s", error)
            return {"status": "error", "error": error}

    @status_update("I'm checking Redis cluster info.")
    async def cluster_info(self) -> Dict[str, Any]:
        """执行 CLUSTER INFO，集群模式不可用时返回结构化错误。"""
        logger.info("Executing CLUSTER INFO")
        try:
            client = self.get_client()
            result = await client.cluster("INFO")

            return {"status": "success", "cluster_info": result}
        except Exception as exc:
            error = _safe_error_message(exc)
            logger.error("Failed to execute CLUSTER INFO: %s", error)
            return {
                "status": "error",
                "error": error,
                "note": "This command only works in cluster mode",
            }

    @status_update("I'm checking Redis replication info.")
    async def replication_info(self) -> Dict[str, Any]:
        """读取 INFO replication 和 ROLE，形成复制状态 evidence。"""
        logger.info("Getting replication info")
        try:
            client = self.get_client()
            info = await client.info("replication")
            role = await client.execute_command("ROLE")

            return {
                "status": "success",
                "info": info,
                "role": {
                    "type": role[0] if role else "unknown",
                    "details": role[1:] if (role and len(role) > 1) else [],
                },
            }
        except Exception as exc:
            error = _safe_error_message(exc)
            logger.error("Failed to get replication info: %s", error)
            return {"status": "error", "error": error}

    @status_update("I'm collecting detailed Redis memory stats.")
    async def memory_stats(self) -> Dict[str, Any]:
        """执行 MEMORY STATS，保留解释说明以免误把字节数当客户端数量。"""
        logger.info("Executing MEMORY STATS")
        try:
            client = self.get_client()
            result = await client.memory_stats()

            return {
                "status": "success",
                "stats": result,
                "interpretation_notes": [
                    "MEMORY STATS reports memory-accounting metrics, not connection counts.",
                    "Fields such as clients.normal and clients.slaves are client-related memory overhead in bytes.",
                    "Use INFO clients or CLIENT LIST to measure current connected clients.",
                ],
                "canonical_sources": {
                    "memory_breakdown": ["MEMORY STATS", "INFO memory"],
                    "client_counts": ["INFO clients", "CLIENT LIST"],
                },
            }
        except Exception as exc:
            error = _safe_error_message(exc)
            if _is_command_unavailable_error(exc):
                logger.warning("MEMORY STATS unavailable on this Redis instance: %s", error)
                return {
                    "status": "error",
                    "error": error,
                    "error_type": "unsupported_command",
                }

            logger.error("Failed to execute MEMORY STATS: %s", error)
            return {"status": "error", "error": error}

    @status_update("I'm scanning Redis for large keys within a bounded safety budget.")
    async def bigkey_scan(
        self,
        threshold_bytes: int = 1024 * 1024,
        max_keys: int = 10_000,
        scan_count: int = 500,
        top_n: int = 20,
        time_limit_ms: int = 5_000,
    ) -> Dict[str, Any]:
        """Find memory-heavy keys without blocking Redis with KEYS."""
        try:
            threshold = max(0, int(threshold_bytes))
            key_budget = max(1, min(int(max_keys), 100_000))
            count_hint = max(1, min(int(scan_count), 2_000))
            result_limit = max(1, min(int(top_n), 100))
            time_budget_ms = max(100, min(int(time_limit_ms), 30_000))
        except (TypeError, ValueError) as exc:
            return {
                "status": "error",
                "error_type": "invalid_parameters",
                "error": _safe_error_message(exc),
            }

        logger.info(
            "Scanning for big keys (threshold_bytes=%s, max_keys=%s, time_limit_ms=%s)",
            threshold,
            key_budget,
            time_budget_ms,
        )
        client = self.get_client()
        started = time.monotonic()
        cursor = 0
        seen: set[str] = set()
        keys_measured = 0
        big_key_count = 0
        sequence = 0
        largest_heap: list[tuple[int, int, Dict[str, Any]]] = []
        big_heap: list[tuple[int, int, Dict[str, Any]]] = []
        stop_reason = "cursor_exhausted"
        scan_complete = False

        def retain_largest(
            heap: list[tuple[int, int, Dict[str, Any]]],
            item: Dict[str, Any],
        ) -> None:
            nonlocal sequence
            sequence += 1
            entry = (item["memory_bytes"], sequence, item)
            if len(heap) < result_limit:
                heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]:
                heapq.heapreplace(heap, entry)

        try:
            while True:
                if (time.monotonic() - started) * 1000 >= time_budget_ms:
                    stop_reason = "time_limit_reached"
                    break

                cursor, keys = await client.scan(cursor=cursor, count=count_hint)
                fresh_keys: list[str] = []
                budget_hit = False
                for raw_key in keys:
                    key = str(raw_key)
                    if key in seen:
                        continue
                    if len(seen) >= key_budget:
                        budget_hit = True
                        break
                    seen.add(key)
                    fresh_keys.append(key)

                if fresh_keys:
                    pipe = client.pipeline(transaction=False)
                    for key in fresh_keys:
                        pipe.type(key)
                        pipe.memory_usage(key, samples=5)
                    measurements = await pipe.execute()

                    for index, key in enumerate(fresh_keys):
                        key_type = measurements[index * 2]
                        memory_bytes = measurements[index * 2 + 1]
                        if memory_bytes is None:
                            continue
                        size = int(memory_bytes)
                        keys_measured += 1
                        item = {
                            "key": key,
                            "type": str(key_type),
                            "memory_bytes": size,
                            "is_big": size >= threshold,
                        }
                        retain_largest(largest_heap, item)
                        if item["is_big"]:
                            big_key_count += 1
                            retain_largest(big_heap, item)

                if budget_hit:
                    stop_reason = "max_keys_reached"
                    break
                if int(cursor) == 0:
                    scan_complete = True
                    stop_reason = "cursor_exhausted"
                    break
                if len(seen) >= key_budget:
                    stop_reason = "max_keys_reached"
                    break

            largest_keys = [entry[2] for entry in sorted(largest_heap, reverse=True)]
            big_keys = [entry[2] for entry in sorted(big_heap, reverse=True)]
            return {
                "status": "success",
                "threshold_bytes": threshold,
                "keys_scanned": len(seen),
                "keys_measured": keys_measured,
                "big_key_count": big_key_count,
                "largest_keys": largest_keys,
                "big_keys": big_keys,
                "scan_complete": scan_complete,
                "stop_reason": stop_reason,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                "limits": {
                    "max_keys": key_budget,
                    "scan_count": count_hint,
                    "top_n": result_limit,
                    "time_limit_ms": time_budget_ms,
                },
            }
        except Exception as exc:
            error = _safe_error_message(exc)
            if _is_command_unavailable_error(exc):
                logger.warning("Big-key scan unavailable on this Redis instance: %s", error)
                return {
                    "status": "error",
                    "error": error,
                    "error_type": "unsupported_command",
                }
            logger.error("Failed to scan Redis for big keys: %s", error)
            return {"status": "error", "error": error}

    @status_update("I'm sampling random keys from the Redis keyspace (count: {count}).")
    async def sample_keys(self, count: int = 100) -> Dict[str, Any]:
        """用 RANDOMKEY 和 TYPE 轻量采样 key，保留数量上限和时间上限。"""
        logger.info("Sampling %s random keys", count)
        try:
            client = self.get_client()

            max_count = 200
            time_limit_secs = 1.0
            batch_attempts_max = 100
            attempt_factor = 3

            try:
                requested = int(count)
            except Exception:
                requested = 100
            target = max(0, min(requested, max_count))

            start = time.monotonic()
            sampled: Dict[str, str] = {}
            attempts = 0
            batches = 0
            max_attempts_total = max(50, target * 5)

            while len(sampled) < target and (time.monotonic() - start) < time_limit_secs:
                remaining = target - len(sampled)
                to_attempt = min(max(remaining * attempt_factor, 10), batch_attempts_max)
                if attempts >= max_attempts_total:
                    break
                to_attempt = min(to_attempt, max_attempts_total - attempts)

                pipe = client.pipeline(transaction=False)
                for _ in range(int(to_attempt)):
                    pipe.randomkey()
                keys = await pipe.execute()

                fresh: List[str] = []
                seen_batch = set()
                for key in keys:
                    if not key:
                        continue
                    if key in sampled or key in seen_batch:
                        continue
                    seen_batch.add(key)
                    fresh.append(key)

                if fresh:
                    pipe2 = client.pipeline(transaction=False)
                    for key in fresh:
                        pipe2.type(key)
                    types = await pipe2.execute()
                    for key, key_type in zip(fresh, types):
                        if len(sampled) >= target:
                            break
                        sampled[key] = key_type

                attempts += int(to_attempt)
                batches += 1

            items = [{"key": key, "type": value} for key, value in list(sampled.items())[:target]]
            type_counts: Dict[str, int] = {}
            for item in items:
                key_type = item["type"]
                type_counts[key_type] = type_counts.get(key_type, 0) + 1

            limit_applied = (requested > max_count) or (
                (time.monotonic() - start) >= time_limit_secs
            )
            return {
                "status": "success",
                "requested_count": requested,
                "sampled_count": len(items),
                "keys": items,
                "type_distribution": type_counts,
                "limit_applied": bool(limit_applied),
                "attempts": attempts,
                "batches": batches,
            }
        except Exception as exc:
            error = _safe_error_message(exc)
            logger.error("Failed to sample keys: %s", error)
            return {"status": "error", "error": error}

    @status_update("I'm listing search indexes.")
    async def search_indexes(self) -> Dict[str, Any]:
        """执行 FT._LIST，列出 Redis Search 索引。"""
        logger.info("Listing Redis Search indexes")
        try:
            client = self.get_client()
            result = await client.execute_command("FT._LIST")
            if result is None:
                indexes = []
            else:
                indexes = [
                    index.decode("utf-8", errors="replace") if isinstance(index, bytes) else index
                    for index in result
                ]

            return {"status": "success", "count": len(indexes), "indexes": indexes}
        except Exception as exc:
            error = _safe_error_message(exc)
            logger.error("Failed to list search indexes: %s", error)
            return {
                "status": "error",
                "error": error,
                "note": "This command requires RediSearch module to be loaded",
            }

    @status_update("I'm getting search index info for {index_name}.")
    async def search_index_info(self, index_name: str) -> Dict[str, Any]:
        """执行 FT.INFO，并把平铺的 key-value 数组转成 dict。"""
        logger.info("Getting info for search index: %s", index_name)
        try:
            client = self.get_client()
            result = await client.execute_command("FT.INFO", index_name)

            info = {}
            if result is not None:
                index = 0
                while index + 1 < len(result):
                    key = result[index].decode("utf-8", errors="replace") if isinstance(result[index], bytes) else result[index]
                    value = result[index + 1]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    elif isinstance(value, list):
                        value = [
                            item.decode("utf-8", errors="replace") if isinstance(item, bytes) else item
                            for item in value
                        ]
                    info[key] = value
                    index += 2

            return {"status": "success", "index_name": index_name, "info": info}
        except Exception as exc:
            error = _safe_error_message(exc)
            logger.error("Failed to get search index info: %s", error)
            return {
                "status": "error",
                "error": error,
                "index_name": index_name,
                "note": "This command requires RediSearch module and a valid index name",
            }

    async def system_hosts(self) -> List[SystemHost]:
        """从 Redis cluster、replication 或连接信息推断系统主机。"""
        hosts: dict[Tuple[str, Optional[int]], SystemHost] = {}
        client = self.get_client()

        def add_host(
            host: str,
            port: Optional[int] = None,
            role: Optional[str] = None,
            labels: Optional[Dict[str, str]] = None,
        ) -> None:
            if not host:
                return
            key = (host, port)
            if key not in hosts:
                hosts[key] = SystemHost(host=host, port=port, role=role, labels=labels or {})
            else:
                if role and not hosts[key].role:
                    hosts[key].role = role
                if labels:
                    hosts[key].labels.update(labels)

        try:
            nodes_raw = await client.cluster("NODES")
            if isinstance(nodes_raw, (bytes, bytearray)):
                nodes_raw = nodes_raw.decode("utf-8", errors="replace")
            if isinstance(nodes_raw, str) and nodes_raw.strip():
                for line in nodes_raw.splitlines():
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    address = parts[1]
                    flags = parts[2]
                    host_port = address.split("@")[0]
                    if host_port.startswith("[") and "]" in host_port:
                        host = host_port[1 : host_port.rfind("]")]
                        port_str = host_port[host_port.rfind("]") + 2 :]
                    else:
                        split_host_port = host_port.rsplit(":", 1)
                        if len(split_host_port) == 2:
                            host, port_str = split_host_port[0], split_host_port[1]
                        else:
                            host, port_str = host_port, None
                    try:
                        port = int(port_str) if port_str else None
                    except Exception:
                        port = None

                    role = None
                    lowered_flags = flags.lower()
                    if "master" in lowered_flags:
                        role = "cluster-master"
                    elif "replica" in lowered_flags or "slave" in lowered_flags:
                        role = "cluster-replica"
                    add_host(host, port, role, labels={"source": "cluster"})
        except Exception:
            pass

        if hosts:
            return list(hosts.values())

        try:
            replication = await client.info("replication")
            if isinstance(replication, dict):
                master_host = replication.get("master_host") or replication.get("master_host_ip")
                master_port = None
                try:
                    master_port = (
                        int(replication.get("master_port"))
                        if replication.get("master_port")
                        else None
                    )
                except Exception:
                    master_port = None
                if master_host:
                    add_host(
                        str(master_host),
                        master_port,
                        role="master",
                        labels={"source": "replication"},
                    )

                for key, value in replication.items():
                    if not isinstance(key, str) or not key.startswith("slave"):
                        continue
                    replica_host = None
                    replica_port = None
                    if isinstance(value, dict):
                        replica_host = value.get("ip") or value.get("host")
                        try:
                            replica_port = int(value.get("port")) if value.get("port") else None
                        except Exception:
                            replica_port = None
                    elif isinstance(value, str):
                        try:
                            match_ip = re.search(r"ip=([^,\s]+)", value)
                            match_port = re.search(r"port=(\d+)", value)
                            replica_host = match_ip.group(1) if match_ip else None
                            replica_port = int(match_port.group(1)) if match_port else None
                        except Exception:
                            replica_host, replica_port = None, None
                    if replica_host:
                        add_host(
                            str(replica_host),
                            replica_port,
                            role="replica",
                            labels={"source": "replication"},
                        )
        except Exception:
            pass

        if not hosts:
            try:
                client_id = await client.client_id()
                client_list = await client.client_list()
                entry = None
                for row in client_list or []:
                    try:
                        if int(row.get("id")) == int(client_id):
                            entry = row
                            break
                    except Exception:
                        continue
                if entry:
                    local_address = entry.get("laddr")
                    if isinstance(local_address, str) and ":" in local_address:
                        host, port_text = local_address.rsplit(":", 1)
                        try:
                            add_host(
                                host,
                                int(port_text),
                                role="single",
                                labels={"source": "client_list"},
                            )
                        except Exception:
                            add_host(host, None, role="single", labels={"source": "client_list"})
            except Exception:
                pass

        if not hosts:
            try:
                parsed = urlparse(self.connection_url)
                if parsed.hostname:
                    port_value = None
                    try:
                        port_value = int(parsed.port) if parsed.port else None
                    except Exception:
                        port_value = None
                    add_host(
                        parsed.hostname,
                        port_value,
                        role="single",
                        labels={"source": "connection_url"},
                    )
            except Exception:
                pass

        return list(hosts.values())
