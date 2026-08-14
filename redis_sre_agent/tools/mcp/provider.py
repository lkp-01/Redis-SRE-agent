"""把受信任配置中的外部 MCP Server 接入现有 ToolManager。

每个 provider 只活在当前 ToolManager 生命周期内。远端目录、schema 和结果都按不可信
输入处理；只有配置 allowlist 中显式标记为 READ 的工具会暴露给 Agent。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from redis_sre_agent.core.config import MCPServerConfig, MCPToolConfig, settings
from redis_sre_agent.tools.models import (
    Tool,
    ToolActionKind,
    ToolCapability,
    ToolDefinition,
    ToolMetadata,
)
from redis_sre_agent.tools.protocols import ToolProvider

if TYPE_CHECKING:
    from redis_sre_agent.core.instances import RedisInstance

logger = logging.getLogger(__name__)

MCP_MAX_DESCRIPTION_CHARS = 4000
MCP_MAX_SCHEMA_BYTES = 65536
MCP_MAX_RESULT_CHARS = 32000
MCP_TOOL_NAME_MAX_CHARS = 64

_PLACEHOLDER_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")
_WINDOWS_CHILD_ENV_KEYS = ("SYSTEMROOT", "COMSPEC", "TEMP", "TMP")


class MCPProviderError(RuntimeError):
    """只携带稳定错误码，不保存远端异常原文。"""


def _coerce_input_schema_dict(input_schema: Any) -> Optional[Dict[str, Any]]:
    """把 MCP SDK/model-like schema 规范化为普通 JSON dict。"""

    if isinstance(input_schema, dict):
        return dict(input_schema)
    if hasattr(input_schema, "model_dump"):
        try:
            payload = input_schema.model_dump(mode="json")
        except TypeError:
            payload = input_schema.model_dump()
        if isinstance(payload, dict):
            return dict(payload)
    if hasattr(input_schema, "dict"):
        payload = input_schema.dict()
        if isinstance(payload, dict):
            return dict(payload)
    if hasattr(input_schema, "items"):
        try:
            return dict(input_schema.items())
        except Exception:
            return None
    return None


def _slug(value: str, *, fallback: str) -> str:
    normalized = _SLUG_RE.sub("_", str(value or "")).strip("_")
    return normalized or fallback


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class MCPToolProvider(ToolProvider):
    """连接一个 MCP Server，并生成受限的本地 Tool 对象。"""

    DEFAULT_CAPABILITY = ToolCapability.UTILITIES

    def __init__(
        self,
        server_name: str,
        server_config: MCPServerConfig,
        redis_instance: Optional["RedisInstance"] = None,
    ) -> None:
        super().__init__(redis_instance=redis_instance)
        self._server_name = str(server_name)
        self._server_slug = _slug(self._server_name, fallback="server")
        self._server_config = server_config
        self._instance_hash = hashlib.sha256(self._server_name.encode("utf-8")).hexdigest()[:6]
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._mcp_tools: List[Any] = []
        self._tool_cache: List[Tool] = []
        self._tools_built = False
        self._tool_name_to_remote: Dict[str, str] = {}
        self._remote_to_tool_name: Dict[str, str] = {}
        self._warned_missing_allowlist = False

    @property
    def provider_name(self) -> str:
        return f"mcp_{self._server_slug}"

    @property
    def requires_redis_instance(self) -> bool:
        return False

    async def __aenter__(self) -> "MCPToolProvider":
        await self._connect()
        return self

    async def __aexit__(self, *_args) -> None:
        await self._disconnect()

    @staticmethod
    def _timeout_seconds() -> float:
        try:
            value = float(settings.tool_timeout)
        except (TypeError, ValueError):
            value = 60.0
        return max(value, 0.001)

    @staticmethod
    def _expand_config_value(value: str) -> str:
        """只展开 `${VAR}`；缺少变量时以稳定错误码拒绝连接。"""

        def replace(match: re.Match[str]) -> str:
            name = match.group(0)[2:-1]
            if name not in os.environ:
                raise MCPProviderError("mcp_connect_failed")
            return os.environ[name]

        expanded = _PLACEHOLDER_RE.sub(replace, str(value))
        if _PLACEHOLDER_RE.search(expanded):
            raise MCPProviderError("mcp_connect_failed")
        return expanded

    def _build_stdio_environment(self) -> Dict[str, str]:
        child_env: Dict[str, str] = {}
        required_keys = ("PATH", *_WINDOWS_CHILD_ENV_KEYS) if os.name == "nt" else ("PATH",)
        for key in required_keys:
            value = os.environ.get(key)
            if value is not None:
                child_env[key] = value
        for key, value in (self._server_config.env or {}).items():
            child_env[str(key)] = self._expand_config_value(value)
        return child_env

    def _expanded_headers(self) -> Optional[Dict[str, str]]:
        if not self._server_config.headers:
            return None
        return {
            str(key): self._expand_config_value(value)
            for key, value in self._server_config.headers.items()
        }

    async def _enter_async_context(self, context_manager: Any) -> Any:
        if self._exit_stack is None:
            raise MCPProviderError("mcp_connect_failed")
        return await asyncio.wait_for(
            self._exit_stack.enter_async_context(context_manager),
            timeout=self._timeout_seconds(),
        )

    async def initialize(self) -> Any:
        if self._session is None:
            raise MCPProviderError("mcp_connect_failed")
        try:
            return await asyncio.wait_for(
                self._session.initialize(),
                timeout=self._timeout_seconds(),
            )
        except TimeoutError as exc:
            raise MCPProviderError("mcp_timeout") from None
        except MCPProviderError:
            raise
        except Exception as exc:
            raise MCPProviderError("mcp_connect_failed") from None

    async def list_tools(self) -> List[Any]:
        if self._session is None:
            raise MCPProviderError("mcp_connect_failed")
        try:
            result = await asyncio.wait_for(
                self._session.list_tools(),
                timeout=self._timeout_seconds(),
            )
        except TimeoutError:
            raise MCPProviderError("mcp_timeout") from None
        except MCPProviderError:
            raise
        except Exception:
            raise MCPProviderError("mcp_discovery_failed") from None

        tools = getattr(result, "tools", None)
        if not isinstance(tools, list):
            raise MCPProviderError("mcp_discovery_failed")
        self._mcp_tools = list(tools)
        self._tool_cache = []
        self._tools_built = False
        self._tool_name_to_remote.clear()
        self._remote_to_tool_name.clear()
        return self._mcp_tools

    async def _connect(self) -> None:
        if self._session is not None:
            return

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        try:
            if self._server_config.command:
                server_params = StdioServerParameters(
                    command=self._server_config.command,
                    args=self._server_config.args or [],
                    env=self._build_stdio_environment(),
                )
                errlog = self._exit_stack.enter_context(
                    open(os.devnull, "w", encoding="utf-8")
                )
                read_stream, write_stream = await self._enter_async_context(
                    stdio_client(server_params, errlog=errlog)
                )
            elif self._server_config.url:
                url = self._expand_config_value(self._server_config.url)
                headers = self._expanded_headers()
                transport = self._server_config.transport or "streamable_http"
                if transport == "sse":
                    read_stream, write_stream = await self._enter_async_context(
                        sse_client(
                            url,
                            headers=headers,
                            timeout=self._timeout_seconds(),
                            sse_read_timeout=self._timeout_seconds(),
                        )
                    )
                else:
                    read_stream, write_stream, _get_session_id = await self._enter_async_context(
                        streamablehttp_client(
                            url,
                            headers=headers,
                            timeout=self._timeout_seconds(),
                            sse_read_timeout=self._timeout_seconds(),
                        )
                    )
            else:
                raise MCPProviderError("mcp_connect_failed")

            session = ClientSession(read_stream, write_stream)
            self._session = await self._enter_async_context(session)
            await self.initialize()
            await self.list_tools()
        except asyncio.CancelledError:
            await self._close_owned_stack()
            raise
        except MCPProviderError as exc:
            code = str(exc)
            await self._close_owned_stack()
            logger.warning("External MCP provider unavailable: %s", code)
            raise
        except TimeoutError:
            await self._close_owned_stack()
            logger.warning("External MCP provider unavailable: mcp_timeout")
            raise MCPProviderError("mcp_timeout") from None
        except Exception:
            await self._close_owned_stack()
            logger.warning("External MCP provider unavailable: mcp_connect_failed")
            raise MCPProviderError("mcp_connect_failed") from None

    async def _close_owned_stack(self) -> None:
        stack = self._exit_stack
        self._exit_stack = None
        try:
            if stack is not None:
                await asyncio.wait_for(stack.aclose(), timeout=self._timeout_seconds())
        except TimeoutError:
            logger.warning("External MCP provider close failed: mcp_timeout")
        except Exception:
            logger.warning("External MCP provider close failed: mcp_connect_failed")
        finally:
            self._session = None

    async def _disconnect(self) -> None:
        try:
            await self._close_owned_stack()
        finally:
            self._session = None
            self._mcp_tools = []
            self._tool_cache = []
            self._tools_built = False
            self._tool_name_to_remote.clear()
            self._remote_to_tool_name.clear()

    def _get_tool_config(self, tool_name: str) -> Optional[MCPToolConfig]:
        tools = self._server_config.tools or {}
        return tools.get(tool_name)

    def _should_include_tool(self, tool_name: str) -> bool:
        tools = self._server_config.tools
        if not tools:
            if not self._warned_missing_allowlist:
                logger.warning("External MCP provider exposes no tools: allowlist_required")
                self._warned_missing_allowlist = True
            return False
        config = tools.get(tool_name)
        return bool(config and config.action_kind is ToolActionKind.READ)

    def _get_capability(self, tool_name: str) -> ToolCapability:
        config = self._get_tool_config(tool_name)
        return config.capability if config and config.capability else self.DEFAULT_CAPABILITY

    def _get_description(self, tool_name: str, remote_description: str) -> str:
        config = self._get_tool_config(tool_name)
        description = remote_description
        if config and config.description:
            description = config.description.replace("{original}", remote_description)
        return str(description)[:MCP_MAX_DESCRIPTION_CHARS]

    def _get_action_kind(self, tool_name: str) -> ToolActionKind:
        config = self._get_tool_config(tool_name)
        if config and config.action_kind is ToolActionKind.READ:
            return ToolActionKind.READ
        return ToolActionKind.UNKNOWN

    def _make_tool_name(self, operation: str) -> str:
        operation_slug = _slug(operation, fallback="tool")
        candidate = f"{self.provider_name}_{self._instance_hash}_{operation_slug}"
        if len(candidate) <= MCP_TOOL_NAME_MAX_CHARS:
            return candidate
        suffix = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:8]
        prefix_length = MCP_TOOL_NAME_MAX_CHARS - len(suffix) - 1
        return f"{candidate[:prefix_length]}_{suffix}"

    def resolve_operation(self, tool_name: str, _args: Dict[str, Any]) -> Optional[str]:
        return self._tool_name_to_remote.get(tool_name)

    @staticmethod
    def _normalize_parameters(input_schema: Any) -> Dict[str, Any]:
        schema = _coerce_input_schema_dict(input_schema)
        if schema is None:
            raise MCPProviderError("mcp_discovery_failed")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise MCPProviderError("mcp_discovery_failed")
        if any(not isinstance(name, str) for name in required):
            raise MCPProviderError("mcp_discovery_failed")
        try:
            parameters = json.loads(
                json.dumps(
                    {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                    ensure_ascii=False,
                )
            )
            encoded = _compact_json(parameters).encode("utf-8")
        except (TypeError, ValueError, OverflowError):
            raise MCPProviderError("mcp_discovery_failed") from None
        if len(encoded) > MCP_MAX_SCHEMA_BYTES:
            raise MCPProviderError("mcp_discovery_failed")
        return parameters

    def _validate_discovered_names(self) -> None:
        seen_remote: set[str] = set()
        seen_slugs: set[str] = set()
        for remote_tool in self._mcp_tools:
            name = getattr(remote_tool, "name", None)
            if not isinstance(name, str) or not name:
                raise MCPProviderError("mcp_discovery_failed")
            slug = _slug(name, fallback="tool")
            if name in seen_remote or slug in seen_slugs:
                raise MCPProviderError("mcp_discovery_failed")
            seen_remote.add(name)
            seen_slugs.add(slug)

    def create_tool_schemas(self) -> List[ToolDefinition]:
        if not self._server_config.tools:
            self._should_include_tool("")
            return []
        self._validate_discovered_names()
        schemas: List[ToolDefinition] = []
        generated_names: set[str] = set()
        self._tool_name_to_remote.clear()
        self._remote_to_tool_name.clear()

        for remote_tool in self._mcp_tools:
            remote_name = remote_tool.name
            if not self._should_include_tool(remote_name):
                continue
            remote_description = getattr(remote_tool, "description", None)
            if not isinstance(remote_description, str) or not remote_description:
                remote_description = f"MCP tool: {remote_name}"
            name = self._make_tool_name(remote_name)
            if name in generated_names:
                raise MCPProviderError("mcp_discovery_failed")
            generated_names.add(name)
            self._tool_name_to_remote[name] = remote_name
            self._remote_to_tool_name[remote_name] = name
            schemas.append(
                ToolDefinition(
                    name=name,
                    description=self._get_description(remote_name, remote_description),
                    capability=self._get_capability(remote_name),
                    parameters=self._normalize_parameters(
                        getattr(remote_tool, "inputSchema", None)
                    ),
                )
            )
        return schemas

    def get_input_schemas(self) -> Dict[str, Dict[str, Any]]:
        schemas: Dict[str, Dict[str, Any]] = {}
        for remote_tool in self._mcp_tools:
            name = getattr(remote_tool, "name", None)
            if not isinstance(name, str) or not self._should_include_tool(name):
                continue
            schema = _coerce_input_schema_dict(getattr(remote_tool, "inputSchema", None))
            if schema is not None:
                schemas[name] = schema
        return schemas

    def tools(self) -> List[Tool]:
        if self._tools_built:
            return self._tool_cache
        schemas = self.create_tool_schemas()
        tools: List[Tool] = []
        for schema in schemas:
            remote_name = self._tool_name_to_remote[schema.name]
            metadata = ToolMetadata(
                name=schema.name,
                description=schema.description,
                capability=schema.capability,
                provider_name=self.provider_name,
                requires_instance=False,
                action_kind=self._get_action_kind(remote_name),
            )

            async def invoke(
                args: Dict[str, Any],
                _remote_name: str = remote_name,
            ) -> Any:
                return await self.call_tool(_remote_name, args)

            tools.append(Tool(metadata=metadata, definition=schema, invoke=invoke))
        self._tool_cache = tools
        self._tools_built = True
        return tools

    @staticmethod
    def _bounded_structured_content(value: Dict[str, Any]) -> tuple[Any, int]:
        try:
            rendered = _compact_json(value)
        except (TypeError, ValueError, OverflowError):
            raise MCPProviderError("mcp_invalid_response") from None
        if len(rendered) <= MCP_MAX_RESULT_CHARS:
            return value, len(rendered)

        preview_budget = max(0, MCP_MAX_RESULT_CHARS - 40)
        bounded: Dict[str, Any] = {
            "truncated": True,
            "preview": rendered[:preview_budget],
        }
        bounded_rendered = _compact_json(bounded)
        while len(bounded_rendered) > MCP_MAX_RESULT_CHARS and bounded["preview"]:
            overflow = len(bounded_rendered) - MCP_MAX_RESULT_CHARS
            bounded["preview"] = bounded["preview"][:-max(1, overflow)]
            bounded_rendered = _compact_json(bounded)
        return bounded, len(bounded_rendered)

    @classmethod
    def _format_result(cls, result: Any) -> Dict[str, Any]:
        if bool(getattr(result, "isError", False)):
            return {"status": "error", "error": "mcp_tool_error"}
        content = getattr(result, "content", None)
        structured = getattr(result, "structuredContent", None)
        if not isinstance(content, list) or (structured is not None and not isinstance(structured, dict)):
            raise MCPProviderError("mcp_invalid_response")

        response: Dict[str, Any] = {"status": "success"}
        used_chars = 0
        if structured is not None:
            bounded, used_chars = cls._bounded_structured_content(structured)
            response["data"] = bounded

        text_parts: List[str] = []
        metadata: Dict[str, int] = {}
        for item in content:
            content_type = getattr(item, "type", None)
            if content_type == "text":
                text = getattr(item, "text", None)
                if not isinstance(text, str):
                    raise MCPProviderError("mcp_invalid_response")
                text_parts.append(text)
            elif content_type in {"image", "audio", "resource", "resource_link"}:
                key = "resource" if content_type == "resource_link" else content_type
                metadata[key] = metadata.get(key, 0) + 1
            else:
                metadata["other"] = metadata.get("other", 0) + 1

        if text_parts:
            remaining = max(0, MCP_MAX_RESULT_CHARS - used_chars)
            response["text"] = "\n".join(text_parts)[:remaining]
        if metadata:
            response["content_metadata"] = metadata
        return response

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """调用远端原始工具名，并把结果压缩成安全的本地边界。"""

        if self._session is None:
            return {"status": "error", "error": "mcp_connect_failed"}
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments=dict(args or {})),
                timeout=self._timeout_seconds(),
            )
            return self._format_result(result)
        except TimeoutError:
            logger.warning("External MCP tool failed: mcp_timeout")
            return {"status": "error", "error": "mcp_timeout"}
        except MCPProviderError as exc:
            code = str(exc)
            logger.warning("External MCP tool failed: %s", code)
            return {"status": "error", "error": code}
        except Exception:
            logger.warning("External MCP tool failed: mcp_tool_error")
            return {"status": "error", "error": "mcp_tool_error"}

    async def _call_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """保留 original 的内部方法名，执行仍委托给同一安全入口。"""

        return await self.call_tool(tool_name, args)
