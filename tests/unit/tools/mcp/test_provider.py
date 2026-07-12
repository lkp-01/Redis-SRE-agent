"""外部 MCP provider 的只读暴露契约。"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from mcp import types as mcp_types

from redis_sre_agent.core.config import MCPServerConfig, MCPToolConfig
from redis_sre_agent.tools.models import ToolActionKind, ToolCapability


def _provider_class():
    module = importlib.import_module("redis_sre_agent.tools.mcp.provider")
    return module.MCPToolProvider


def test_provider_exposes_only_explicit_read_allowlist_with_stable_safe_name() -> None:
    """远端目录不可信，只有显式标记为 READ 的 allowlist 项可以进入 Agent。"""

    provider_class = _provider_class()
    config = MCPServerConfig(
        command="fake-mcp-command",
        tools={
            "read status": MCPToolConfig(
                capability=ToolCapability.DIAGNOSTICS,
                description="读取外部系统状态。",
                action_kind=ToolActionKind.READ,
            ),
            "write_status": MCPToolConfig(action_kind=ToolActionKind.WRITE),
            "unknown_status": MCPToolConfig(),
        },
    )
    remote_tools = [
        SimpleNamespace(
            name="read status",
            description="Read status from the external system.",
            inputSchema={
                "type": "object",
                "properties": {"detail": {"type": "boolean"}},
            },
        ),
        SimpleNamespace(
            name="write_status",
            description="Write status to the external system.",
            inputSchema={"type": "object", "properties": {}},
        ),
        SimpleNamespace(
            name="unknown_status",
            description="Perform an unspecified operation.",
            inputSchema={"type": "object", "properties": {}},
        ),
        SimpleNamespace(
            name="not_allowlisted",
            description="Read an unapproved value.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]

    first = provider_class(server_name="Status API!", server_config=config)
    second = provider_class(server_name="Status API!", server_config=config)
    first._mcp_tools = remote_tools
    second._mcp_tools = remote_tools

    first_tools = first.tools()
    second_tools = second.tools()

    assert len(first_tools) == 1
    assert [tool.definition.name for tool in first_tools] == [
        tool.definition.name for tool in second_tools
    ]
    tool = first_tools[0]
    assert tool.metadata.action_kind is ToolActionKind.READ
    assert tool.metadata.capability is ToolCapability.DIAGNOSTICS
    assert tool.definition.description == "读取外部系统状态。"
    assert len(tool.definition.name) <= 64
    assert re.fullmatch(r"[A-Za-z0-9_-]+", tool.definition.name)


def test_provider_without_allowlist_exposes_no_remote_tools() -> None:
    provider_class = _provider_class()
    provider = provider_class(
        server_name="unconfigured",
        server_config=MCPServerConfig(command="fake-mcp-command"),
    )
    provider._mcp_tools = [
        SimpleNamespace(
            name="read_status",
            description="Read status.",
            inputSchema={"type": "object", "properties": {}},
        )
    ]

    assert provider.tools() == []


def test_schema_model_is_coerced_and_description_is_bounded() -> None:
    provider_module = importlib.import_module("redis_sre_agent.tools.mcp.provider")

    class SchemaModel:
        def model_dump(self, mode: str = "json") -> dict[str, object]:
            assert mode == "json"
            return {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": ["integer", "null"]},
                },
                "required": ["name"],
            }

    assert provider_module._coerce_input_schema_dict(SchemaModel()) == {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": ["integer", "null"]},
        },
        "required": ["name"],
    }

    provider = provider_module.MCPToolProvider(
        server_name="schema",
        server_config=MCPServerConfig(
            command="fake-mcp-command",
            tools={"read_status": MCPToolConfig(action_kind=ToolActionKind.READ)},
        ),
    )
    provider._mcp_tools = [
        SimpleNamespace(
            name="read_status",
            description="d" * 5000,
            inputSchema=SchemaModel(),
        )
    ]

    tool = provider.tools()[0]
    assert len(tool.definition.description) == provider_module.MCP_MAX_DESCRIPTION_CHARS
    assert tool.definition.parameters["properties"]["count"]["type"] == ["integer", "null"]
    assert tool.definition.parameters["required"] == ["name"]


@pytest.mark.parametrize(
    "remote_names",
    [
        ["read_status", "read_status"],
        ["read status", "read-status"],
    ],
)
def test_duplicate_remote_or_normalized_names_fail_provider_discovery(remote_names) -> None:
    provider_module = importlib.import_module("redis_sre_agent.tools.mcp.provider")
    provider = provider_module.MCPToolProvider(
        server_name="duplicates",
        server_config=MCPServerConfig(
            command="fake-mcp-command",
            tools={
                name: MCPToolConfig(action_kind=ToolActionKind.READ)
                for name in remote_names
            },
        ),
    )
    provider._mcp_tools = [
        SimpleNamespace(
            name=name,
            description="Read status.",
            inputSchema={"type": "object", "properties": {}},
        )
        for name in remote_names
    ]

    with pytest.raises(provider_module.MCPProviderError, match="mcp_discovery_failed"):
        provider.tools()


def test_oversized_schema_fails_closed_without_echoing_payload() -> None:
    provider_module = importlib.import_module("redis_sre_agent.tools.mcp.provider")
    sentinel = "MCP_SCHEMA_SENTINEL"
    provider = provider_module.MCPToolProvider(
        server_name="large-schema",
        server_config=MCPServerConfig(
            command="fake-mcp-command",
            tools={"read_status": MCPToolConfig(action_kind=ToolActionKind.READ)},
        ),
    )
    provider._mcp_tools = [
        SimpleNamespace(
            name="read_status",
            description="Read status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "string",
                        "description": sentinel * 5000,
                    }
                },
            },
        )
    ]

    with pytest.raises(provider_module.MCPProviderError) as exc_info:
        provider.tools()

    assert str(exc_info.value) == "mcp_discovery_failed"
    assert sentinel not in str(exc_info.value)


class FakeSession:
    def __init__(self, read_stream, write_stream) -> None:
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.initialized = False
        self.closed = False
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *_args) -> None:
        self.closed = True

    async def initialize(self):
        self.initialized = True
        return SimpleNamespace()

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                mcp_types.Tool(
                    name="read_status",
                    description="Read status.",
                    inputSchema={
                        "type": "object",
                        "properties": {"detail": {"type": "boolean"}},
                    },
                )
            ]
        )

    async def call_tool(self, name: str, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="healthy")],
            structuredContent={"healthy": True},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "sse", "streamable_http"])
async def test_three_transports_initialize_discover_call_and_close(
    monkeypatch,
    transport: str,
) -> None:
    provider_module = importlib.import_module("redis_sre_agent.tools.mcp.provider")
    seen: dict[str, object] = {"entered": [], "exited": []}
    sessions: list[FakeSession] = []

    @asynccontextmanager
    async def fake_stdio(params, errlog=None):
        seen["stdio_params"] = params
        seen["stdio_errlog"] = errlog
        seen["entered"].append("stdio")
        try:
            yield "stdio-read", "stdio-write"
        finally:
            seen["exited"].append("stdio")

    @asynccontextmanager
    async def fake_sse(url, headers=None, **_kwargs):
        seen["url"] = url
        seen["headers"] = headers
        seen["entered"].append("sse")
        try:
            yield "sse-read", "sse-write"
        finally:
            seen["exited"].append("sse")

    @asynccontextmanager
    async def fake_streamable(url, headers=None, **_kwargs):
        seen["url"] = url
        seen["headers"] = headers
        seen["entered"].append("streamable_http")
        try:
            yield "http-read", "http-write", lambda: "session-id"
        finally:
            seen["exited"].append("streamable_http")

    def fake_client_session(read_stream, write_stream):
        session = FakeSession(read_stream, write_stream)
        sessions.append(session)
        return session

    monkeypatch.setattr(provider_module, "stdio_client", fake_stdio)
    monkeypatch.setattr(provider_module, "sse_client", fake_sse)
    monkeypatch.setattr(provider_module, "streamablehttp_client", fake_streamable)
    monkeypatch.setattr(provider_module, "ClientSession", fake_client_session)

    if transport == "stdio":
        config = MCPServerConfig(
            command="fake-mcp-command",
            env={"EXPLICIT_MCP_VALUE": "${MCP_PARENT_VALUE}"},
            tools={"read_status": MCPToolConfig(action_kind=ToolActionKind.READ)},
        )
    else:
        config = MCPServerConfig(
            url="http://127.0.0.1:8765/mcp",
            transport=transport,
            headers={"Authorization": "Bearer ${MCP_PARENT_VALUE}"},
            tools={"read_status": MCPToolConfig(action_kind=ToolActionKind.READ)},
        )

    safe_parent_env = {
        "PATH": "test-path",
        "SYSTEMROOT": "test-system-root",
        "COMSPEC": "test-comspec",
        "TEMP": "test-temp",
        "TMP": "test-tmp",
        "MCP_PARENT_VALUE": "expanded-test-value",
        "UNRELATED_SECRET": "must-not-be-forwarded",
    }
    with patch.dict(os.environ, safe_parent_env, clear=True):
        async with provider_module.MCPToolProvider(
            server_name="transport-test",
            server_config=config,
        ) as provider:
            tools = provider.tools()
            result = await tools[0].invoke({"detail": True})

    assert seen["entered"] == [transport]
    assert seen["exited"] == [transport]
    assert sessions and sessions[0].initialized is True and sessions[0].closed is True
    assert sessions[0].calls == [("read_status", {"detail": True})]
    assert result == {
        "status": "success",
        "data": {"healthy": True},
        "text": "healthy",
    }
    if transport == "stdio":
        child_env = seen["stdio_params"].env
        assert child_env["EXPLICIT_MCP_VALUE"] == "expanded-test-value"
        assert child_env["PATH"] == "test-path"
        assert child_env["SYSTEMROOT"] == "test-system-root"
        assert "UNRELATED_SECRET" not in child_env
        assert seen["stdio_errlog"] is not None
    else:
        assert seen["headers"] == {"Authorization": "Bearer expanded-test-value"}


@pytest.mark.asyncio
async def test_unresolved_stdio_env_placeholder_fails_before_spawn(monkeypatch, caplog) -> None:
    provider_module = importlib.import_module("redis_sre_agent.tools.mcp.provider")
    sentinel = "MCP_ENV_SENTINEL_SECRET"

    def forbidden_stdio(*_args, **_kwargs):
        raise AssertionError("unresolved env must fail before subprocess creation")

    monkeypatch.setattr(provider_module, "stdio_client", forbidden_stdio)
    config = MCPServerConfig(
        command="fake-mcp-command",
        env={"MCP_TOKEN": "${MISSING_MCP_TEST_VALUE}"},
        tools={"read_status": MCPToolConfig(action_kind=ToolActionKind.READ)},
    )

    with patch.dict(os.environ, {"PATH": "test-path", "OTHER": sentinel}, clear=True):
        with pytest.raises(provider_module.MCPProviderError) as exc_info:
            await provider_module.MCPToolProvider(
                server_name="missing-env",
                server_config=config,
            )._connect()

    assert str(exc_info.value) == "mcp_connect_failed"
    assert sentinel not in caplog.text
    assert sentinel not in str(exc_info.value)


class ResultSession:
    def __init__(self, result=None, exc: BaseException | None = None) -> None:
        self.result = result
        self.exc = exc
        self.seen_name: str | None = None

    async def call_tool(self, name: str, arguments=None):
        self.seen_name = name
        if self.exc is not None:
            raise self.exc
        return self.result


@pytest.mark.asyncio
async def test_result_boundary_truncates_text_and_hides_binary_content() -> None:
    provider_module = importlib.import_module("redis_sre_agent.tools.mcp.provider")
    binary_sentinel = "MCP_BINARY_SENTINEL"
    result = mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(type="text", text="t" * 40000),
            mcp_types.ImageContent(
                type="image",
                data=binary_sentinel * 100,
                mimeType="image/png",
            ),
            mcp_types.AudioContent(
                type="audio",
                data=binary_sentinel * 100,
                mimeType="audio/wav",
            ),
            mcp_types.EmbeddedResource(
                type="resource",
                resource=mcp_types.BlobResourceContents(
                    uri="file:///fake.bin",
                    mimeType="application/octet-stream",
                    blob=binary_sentinel * 100,
                ),
            ),
        ],
        structuredContent={"payload": "s" * 40000},
    )
    provider = provider_module.MCPToolProvider(
        server_name="result-boundary",
        server_config=MCPServerConfig(
            command="fake-mcp-command",
            tools={"large_result": MCPToolConfig(action_kind=ToolActionKind.READ)},
        ),
    )
    provider._session = ResultSession(result=result)

    response = await provider.call_tool("large_result", {})
    rendered = json.dumps(response, ensure_ascii=False, default=str)
    bounded_content = len(
        json.dumps(response.get("data", {}), ensure_ascii=False, separators=(",", ":"))
    ) + len(response.get("text", ""))

    assert response["status"] == "success"
    assert bounded_content <= provider_module.MCP_MAX_RESULT_CHARS
    assert response["content_metadata"] == {"image": 1, "audio": 1, "resource": 1}
    assert binary_sentinel not in rendered


@pytest.mark.asyncio
async def test_mcp_error_exception_and_timeout_return_stable_safe_codes(
    monkeypatch,
    caplog,
) -> None:
    provider_module = importlib.import_module("redis_sre_agent.tools.mcp.provider")
    sentinel = "MCP_REMOTE_SENTINEL_SECRET"
    provider = provider_module.MCPToolProvider(
        server_name="safe-errors",
        server_config=MCPServerConfig(
            command="fake-mcp-command",
            tools={"read_status": MCPToolConfig(action_kind=ToolActionKind.READ)},
        ),
    )

    provider._session = ResultSession(
        result=mcp_types.CallToolResult(
            isError=True,
            content=[mcp_types.TextContent(type="text", text=sentinel)],
        )
    )
    remote_error = await provider.call_tool("read_status", {})

    provider._session = ResultSession(exc=RuntimeError(sentinel))
    exception_error = await provider.call_tool("read_status", {})

    class SlowSession:
        async def call_tool(self, name: str, arguments=None):
            await asyncio.sleep(10)

    monkeypatch.setattr(provider_module.settings, "tool_timeout", 0.01)
    provider._session = SlowSession()
    timeout_error = await provider.call_tool("read_status", {})

    payload = json.dumps(
        [remote_error, exception_error, timeout_error],
        ensure_ascii=False,
    )
    assert remote_error == {"status": "error", "error": "mcp_tool_error"}
    assert exception_error == {"status": "error", "error": "mcp_tool_error"}
    assert timeout_error == {"status": "error", "error": "mcp_timeout"}
    assert sentinel not in payload
    assert sentinel not in caplog.text


@pytest.mark.asyncio
async def test_cancelled_tool_call_is_not_swallowed() -> None:
    provider_module = importlib.import_module("redis_sre_agent.tools.mcp.provider")
    provider = provider_module.MCPToolProvider(
        server_name="cancelled",
        server_config=MCPServerConfig(
            command="fake-mcp-command",
            tools={"read_status": MCPToolConfig(action_kind=ToolActionKind.READ)},
        ),
    )
    provider._session = ResultSession(exc=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await provider.call_tool("read_status", {})


@pytest.mark.asyncio
async def test_discovery_failure_closes_transport_and_redacts_exception(monkeypatch, caplog) -> None:
    provider_module = importlib.import_module("redis_sre_agent.tools.mcp.provider")
    sentinel = "MCP_DISCOVERY_SENTINEL_SECRET"
    exited = False

    @asynccontextmanager
    async def fake_stdio(_params, errlog=None):
        nonlocal exited
        try:
            yield "read", "write"
        finally:
            exited = True

    class FailingSession(FakeSession):
        async def list_tools(self):
            raise RuntimeError(sentinel)

    monkeypatch.setattr(provider_module, "stdio_client", fake_stdio)
    monkeypatch.setattr(
        provider_module,
        "ClientSession",
        lambda read_stream, write_stream: FailingSession(read_stream, write_stream),
    )
    provider = provider_module.MCPToolProvider(
        server_name="discovery-failure",
        server_config=MCPServerConfig(
            command="fake-mcp-command",
            tools={"read_status": MCPToolConfig(action_kind=ToolActionKind.READ)},
        ),
    )

    with pytest.raises(provider_module.MCPProviderError) as exc_info:
        await provider._connect()

    assert str(exc_info.value) == "mcp_discovery_failed"
    assert exited is True
    assert provider._session is None
    assert sentinel not in caplog.text
    assert sentinel not in str(exc_info.value)


@pytest.mark.asyncio
async def test_cancelled_initialize_closes_owned_transport_before_propagating(
    monkeypatch,
) -> None:
    provider_module = importlib.import_module("redis_sre_agent.tools.mcp.provider")
    exited = False
    sessions: list[FakeSession] = []

    @asynccontextmanager
    async def fake_stdio(_params, errlog=None):
        nonlocal exited
        try:
            yield "read", "write"
        finally:
            exited = True

    class CancelledSession(FakeSession):
        async def initialize(self):
            raise asyncio.CancelledError()

    def fake_client_session(read_stream, write_stream):
        session = CancelledSession(read_stream, write_stream)
        sessions.append(session)
        return session

    monkeypatch.setattr(provider_module, "stdio_client", fake_stdio)
    monkeypatch.setattr(provider_module, "ClientSession", fake_client_session)
    provider = provider_module.MCPToolProvider(
        server_name="cancelled-connect",
        server_config=MCPServerConfig(
            command="fake-mcp-command",
            tools={"read_status": MCPToolConfig(action_kind=ToolActionKind.READ)},
        ),
    )

    with pytest.raises(asyncio.CancelledError):
        await provider._connect()

    assert exited is True
    assert sessions and sessions[0].closed is True
    assert provider._session is None
