"""
不管你是查数据库的工具，还是查日志的工具，你都必须按照统一的规矩来注册、组装，并把最终的工具交给系统。
第一部分是各种protocol
第二部分就是ToolProvider 类
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from .models import SystemHost, Tool, ToolCapability, ToolDefinition, ToolMetadata

if TYPE_CHECKING:
    from redis_sre_agent.core.instances import RedisInstance

    from .manager import ToolManager

logger = logging.getLogger(__name__)


class MetricsProviderProtocol(Protocol):
    async def query(self, query: str) -> Dict[str, Any]: ...

    async def query_range(
        self, query: str, start_time: str, end_time: str, step: Optional[str] = None
    ) -> Dict[str, Any]: ...


class LogsProviderProtocol(Protocol):
    async def query_range(
        self,
        query: str,
        start: str,
        end: str,
        step: Optional[str] = None,
        limit: Optional[int] = None,
        interval: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Dict[str, Any]: ...


@runtime_checkable
class DiagnosticsProviderProtocol(Protocol):
    async def info(self, section: Optional[str] = None) -> Dict[str, Any]: ...

    async def replication_info(self) -> Dict[str, Any]: ...

    async def client_list(self, client_type: Optional[str] = None) -> Dict[str, Any]: ...

    async def system_hosts(self) -> List[SystemHost]: ...


@runtime_checkable
class KnowledgeProviderProtocol(Protocol):
    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
        distance_threshold: Optional[float] = None,
    ) -> Dict[str, Any]: ...


@runtime_checkable
class UtilitiesProviderProtocol(Protocol):
    async def calculator(self, expression: str) -> Dict[str, Any]: ...

# 如果要写一个真实去连 Redis 的工具，或者连外部 API 的工具
# 这些“工具提供者”都必须继承这个模具，在这个模具的基础上填空。
class ToolProvider(ABC):
    """所有工具 provider 的基类。"""

    capabilities: set[ToolCapability] = set()
    instance_config_model: Optional[type[BaseModel]] = None
    extension_namespace: Optional[str] = None
    _manager: Optional["ToolManager"] = None

    def __init__(
        self, redis_instance: Optional["RedisInstance"] = None, config: Optional[Any] = None
    ):
        self.redis_instance = redis_instance
        self.config = config
        self.instance_config: Optional[BaseModel] = None
        if redis_instance is not None:
            import hashlib

            self._instance_hash = hashlib.sha256(redis_instance.id.encode()).hexdigest()[:6]
        else:
            self._instance_hash = hex(id(self))[2:8]
        try:
            self.instance_config = self._load_instance_extension_config()
        except Exception:
            self.instance_config = None

    # 找某个工具的特殊名字extension_namespace
    def _get_extension_namespace(self) -> str:
        try:
            ns = (self.extension_namespace or self.provider_name or "").strip()
            return ns or ""
        except Exception:
            return ""

    # 根据特殊名字extension_namespace寻找其特定扩展配置
    def _load_instance_extension_config(self) -> Optional[BaseModel]:
        """解析实例上的 provider 扩展配置。

        这里只做结构解析，不打印 secret，也不主动连接外部系统。
        """

        if not self.instance_config_model or not self.redis_instance:
            return None
        try:
            ns = self._get_extension_namespace()
            data: Dict[str, Any] = {}
            ext = self.redis_instance.extension_data or {}
            if isinstance(ext.get(ns), dict):
                data.update(ext.get(ns) or {})
            else:
                prefix = f"{ns}."
                for key, value in ext.items():
                    if isinstance(key, str) and key.startswith(prefix):
                        data[key[len(prefix) :]] = value
            secrets = self.redis_instance.extension_secrets or {}
            secrets_ns = secrets.get(ns)
            if isinstance(secrets_ns, dict):
                data.update(secrets_ns)
            else:
                prefix = f"{ns}."
                for key, value in secrets.items():
                    if isinstance(key, str) and key.startswith(prefix):
                        data[key[len(prefix) :]] = value
            return self.instance_config_model.model_validate(data or {})
        except Exception:
            return None

    #
    def resolve_operation(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        try:
            prefix = f"{self.provider_name}_{self._instance_hash}_"
            if tool_name.startswith(prefix):
                return tool_name[len(prefix) :]
            parts = tool_name.split("_")
            if len(parts) >= 3:
                return "_".join(parts[2:])
            logger.warning(
                "resolve_operation falling back to full tool name %r as operation.",
                tool_name,
            )
            return tool_name
        except Exception as exc:
            logger.warning("resolve_operation failed for %r: %s", tool_name, exc)
            return None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """provider 类型名，也是工具名前缀。"""
        ...

    async def __aenter__(self) -> "ToolProvider":
        return self

    async def __aexit__(self, *args) -> None:
        return None

    def _make_tool_name(self, operation: str) -> str:
        return f"{self.provider_name}_{self._instance_hash}_{operation}"

    def create_tool_schemas(self) -> List[ToolDefinition]:
        raise NotImplementedError(
            f"{self.__class__.__name__}.create_tool_schemas() is not implemented; "
            "override create_tool_schemas() or tools()."
        )

    @property
    def requires_redis_instance(self) -> bool:
        return self.redis_instance is not None

    def tools(self) -> List[Tool]:
        schemas = self.create_tool_schemas()
        tools: List[Tool] = []
        for schema in schemas:
            meta = ToolMetadata(
                name=schema.name,
                description=schema.description,
                capability=schema.capability,
                provider_name=self.provider_name,
                requires_instance=self.requires_redis_instance,
            )
            op_name = self.resolve_operation(schema.name, {}) or ""
            method = getattr(self, op_name, None) if op_name else None
            if not callable(method):
                raise RuntimeError(
                    f"Provider {self.__class__.__name__} has no method {op_name!r} "
                    f"for tool {schema.name!r}."
                )

            async def _invoke(args: Dict[str, Any], _method=method) -> Any:
                return await _method(**(args or {}))

            tools.append(Tool(metadata=meta, definition=schema, invoke=_invoke))
        return tools

    def get_status_update(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        try:
            op = self.resolve_operation(tool_name, args)
            if not op:
                return None
            method = self.__dict__.get(op) or type(self).__dict__.get(op)
            template = getattr(method, "_status_update_template", None) if method else None
            if not template:
                return None
            try:
                return template.format(**args)
            except Exception:
                return template
        except Exception:
            return None
