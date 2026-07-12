"""External MCP 与现有 Redis 工具主链的共存契约。"""

from __future__ import annotations

import builtins
import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from redis_sre_agent.core.config import Settings
from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.tools.diagnostics.redis_command.provider import RedisCommandToolProvider
from redis_sre_agent.tools.manager import ToolManager
from redis_sre_agent.tools.models import (
    Tool,
    ToolActionKind,
    ToolCapability,
    ToolDefinition,
    ToolMetadata,
)
from redis_sre_agent.tools.target_discovery.provider import TargetDiscoveryToolProvider


class FakeInfoClient:
    async def info(self, section: str | None = None) -> dict[str, Any]:
        return {"redis_version": "mcp-coexist-fake", "section": section or "all"}

    async def aclose(self) -> None:
        return None


class EmptyTargetHandleStore:
    async def get_records(self, _target_handles):
        return {}


class FakeMCPToolProvider:
    """只替换 transport 层；Manager 的注册、路由和关闭仍走真实代码。"""

    instances: list["FakeMCPToolProvider"] = []
    tool_specs = [
        (
            "mcp_demo_a1b2c3_read_status",
            ToolCapability.DIAGNOSTICS,
            ToolActionKind.READ,
        )
    ]

    def __init__(self, server_name: str, server_config, redis_instance=None) -> None:
        self.server_name = server_name
        self.server_config = server_config
        self.redis_instance = redis_instance
        self.closed = False
        self._manager = None
        self.instances.append(self)

    @property
    def provider_name(self) -> str:
        return f"mcp_{self.server_name}"

    async def __aenter__(self) -> "FakeMCPToolProvider":
        return self

    async def __aexit__(self, *_args) -> None:
        self.closed = True

    def tools(self) -> list[Tool]:
        tools: list[Tool] = []
        for name, capability, action_kind in self.tool_specs:
            async def invoke(
                args: dict[str, Any],
                _name: str = name,
            ) -> dict[str, Any]:
                return {
                    "status": "success",
                    "source": "mcp",
                    "tool": _name,
                    "args": dict(args),
                }

            definition = ToolDefinition(
                name=name,
                description="Read fake external status.",
                capability=capability,
                parameters={
                    "type": "object",
                    "properties": {"scope": {"type": "string"}},
                },
            )
            metadata = ToolMetadata(
                name=name,
                description=definition.description,
                capability=definition.capability,
                provider_name=self.provider_name,
                action_kind=action_kind,
            )
            tools.append(Tool(metadata=metadata, definition=definition, invoke=invoke))
        return tools


def make_instance() -> RedisInstance:
    return RedisInstance(
        id="mcp-coexist-instance",
        name="MCP Coexist Cache",
        connection_url=SecretStr("redis://localhost:6379/0"),
        environment="test",
        usage="cache",
        description="Fake instance for MCP coexistence tests.",
    )


@pytest.mark.asyncio
async def test_empty_mcp_config_does_not_import_transport_or_change_builtin_tools(
    monkeypatch,
) -> None:
    import redis_sre_agent.tools.manager as manager_module

    monkeypatch.setattr(
        manager_module,
        "settings",
        Settings(_env_file=None, rag_enabled=False, mcp_servers={}),
    )
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "redis_sre_agent.tools.mcp.provider" or name.startswith("mcp"):
            raise AssertionError("empty MCP config must not import an MCP transport")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    async with ToolManager() as manager:
        names = [tool.name for tool in manager.get_tools_for_llm()]

    assert any(name.endswith("resolve_redis_targets") for name in names)
    assert not any(name.startswith("mcp_") for name in names)


@pytest.mark.asyncio
async def test_mcp_target_discovery_and_dynamic_redis_tools_share_manager_route(
    monkeypatch,
) -> None:
    import redis_sre_agent.tools.manager as manager_module

    provider_module = importlib.import_module("redis_sre_agent.tools.mcp.provider")
    FakeMCPToolProvider.instances.clear()
    monkeypatch.setattr(provider_module, "MCPToolProvider", FakeMCPToolProvider)
    monkeypatch.setattr(
        manager_module,
        "settings",
        Settings(
            _env_file=None,
            rag_enabled=False,
            mcp_servers={
                "demo": {
                    "command": sys.executable,
                    "tools": {
                        "read_status": {
                            "capability": "diagnostics",
                            "action_kind": "read",
                        }
                    },
                }
            },
        ),
    )
    instance = make_instance()

    async def fake_get_instance_by_id(instance_id: str):
        return instance if instance_id == instance.id else None

    def fake_get_client(self):
        if self._client is None:
            self._client = FakeInfoClient()
        return self._client

    async def fake_resolve_targets(
        self,
        query: str,
        allow_multiple: bool = False,
        max_results: int = 5,
        attach_tools: bool = True,
        preferred_capabilities=None,
    ) -> dict[str, Any]:
        assert query == "MCP Coexist Cache"
        binding = SimpleNamespace(
            target_handle="target_mcp_coexist",
            target_kind="instance",
            resource_id=instance.id,
        )
        await self._manager.attach_bound_targets([binding])
        return {
            "status": "resolved",
            "attached_target_handles": [binding.target_handle],
            "toolset_generation": self._manager.get_toolset_generation(),
        }

    monkeypatch.setattr(manager_module, "get_target_handle_store", lambda: EmptyTargetHandleStore())
    monkeypatch.setattr(manager_module, "get_instance_by_id", fake_get_instance_by_id)
    monkeypatch.setattr(RedisCommandToolProvider, "get_client", fake_get_client)
    monkeypatch.setattr(TargetDiscoveryToolProvider, "resolve_redis_targets", fake_resolve_targets)

    async with ToolManager(thread_id="thread-mcp-coexist") as manager:
        initial_names = [tool.name for tool in manager.get_tools_for_llm()]
        resolve_name = next(name for name in initial_names if name.endswith("resolve_redis_targets"))
        mcp_name = next(name for name in initial_names if name.startswith("mcp_"))
        initial_generation = manager.get_toolset_generation()

        mcp_result = await manager.resolve_tool_call(mcp_name, {"scope": "health"})
        discovery_result = await manager.resolve_tool_call(
            resolve_name,
            {"query": "MCP Coexist Cache"},
        )
        rebound_names = [tool.name for tool in manager.get_tools_for_llm()]
        info_name = next(name for name in rebound_names if name.endswith("info"))
        redis_result = await manager.resolve_tool_call(info_name, {})

        assert mcp_name in rebound_names
        assert manager.get_toolset_generation() > initial_generation

    assert mcp_result["status"] == "success"
    assert discovery_result["status"] == "resolved"
    assert redis_result["status"] == "success"
    assert redis_result["data"]["redis_version"] == "mcp-coexist-fake"
    assert FakeMCPToolProvider.instances
    assert all(provider.closed for provider in FakeMCPToolProvider.instances)


def test_mcp_startup_guard_helpers_match_original_path_rules(tmp_path) -> None:
    import redis_sre_agent.tools.manager as manager_module

    missing_command = tmp_path / "missing-mcp-command.exe"
    missing_script = tmp_path / "missing-server.py"

    assert manager_module._command_is_available(str(missing_command)) is False
    assert manager_module._missing_local_mcp_arg_path([str(missing_script)]) == str(
        missing_script
    )
    assert manager_module._missing_local_mcp_arg_path(
        ["-lc", "cd /work && exec python /work/server.py"]
    ) is None
    assert manager_module._missing_local_mcp_arg_path(
        ["https://example.invalid/server.py"]
    ) is None


@pytest.mark.asyncio
async def test_missing_command_is_skipped_without_stack_trace_or_value_leak(
    monkeypatch,
    caplog,
) -> None:
    import redis_sre_agent.tools.manager as manager_module

    missing_command = "MCP_MISSING_COMMAND_SENTINEL_987654"
    monkeypatch.setattr(
        manager_module,
        "settings",
        Settings(
            _env_file=None,
            rag_enabled=False,
            mcp_servers={"missing": {"command": missing_command}},
        ),
    )

    async with ToolManager() as manager:
        names = [tool.name for tool in manager.get_tools_for_llm()]

    assert any(name.endswith("resolve_redis_targets") for name in names)
    assert not any(name.startswith("mcp_") for name in names)
    assert "mcp_command_unavailable" in caplog.text
    assert "Traceback" not in caplog.text
    assert missing_command not in caplog.text


@pytest.mark.asyncio
async def test_provider_start_failure_does_not_block_builtin_tools_or_leak_exception(
    monkeypatch,
    caplog,
) -> None:
    import redis_sre_agent.tools.manager as manager_module

    provider_module = importlib.import_module("redis_sre_agent.tools.mcp.provider")
    sentinel = "MCP_PROVIDER_START_SENTINEL_SECRET"

    class FailingProvider(FakeMCPToolProvider):
        async def __aenter__(self):
            raise RuntimeError(sentinel)

    monkeypatch.setattr(provider_module, "MCPToolProvider", FailingProvider)
    monkeypatch.setattr(
        manager_module,
        "settings",
        Settings(
            _env_file=None,
            rag_enabled=False,
            mcp_servers={
                "failing": {
                    "command": sys.executable,
                    "tools": {"read_status": {"action_kind": "read"}},
                }
            },
        ),
    )

    async with ToolManager() as manager:
        names = [tool.name for tool in manager.get_tools_for_llm()]

    assert any(name.endswith("resolve_redis_targets") for name in names)
    assert not any(name.startswith("mcp_") for name in names)
    assert "mcp_provider_unavailable" in caplog.text
    assert sentinel not in caplog.text
    assert "Traceback" not in caplog.text


@pytest.mark.asyncio
async def test_mcp_name_conflict_skips_entire_provider_without_overwrite(
    monkeypatch,
    caplog,
) -> None:
    import redis_sre_agent.tools.manager as manager_module

    provider_module = importlib.import_module("redis_sre_agent.tools.mcp.provider")
    builtin_resolve_name = TargetDiscoveryToolProvider().create_tool_schemas()[1].name
    unique_name = "mcp_conflict_a1b2c3_unique_read"
    FakeMCPToolProvider.instances.clear()
    monkeypatch.setattr(
        FakeMCPToolProvider,
        "tool_specs",
        [
            (unique_name, ToolCapability.DIAGNOSTICS, ToolActionKind.READ),
            (builtin_resolve_name, ToolCapability.DIAGNOSTICS, ToolActionKind.READ),
        ],
    )
    monkeypatch.setattr(provider_module, "MCPToolProvider", FakeMCPToolProvider)
    monkeypatch.setattr(
        manager_module,
        "settings",
        Settings(
            _env_file=None,
            rag_enabled=False,
            mcp_servers={
                "conflict": {
                    "command": sys.executable,
                    "tools": {"read_status": {"action_kind": "read"}},
                }
            },
        ),
    )

    async with ToolManager() as manager:
        names = [tool.name for tool in manager.get_tools_for_llm()]
        builtin_provider = manager._routing_table[builtin_resolve_name]

        assert unique_name not in names
        assert names.count(builtin_resolve_name) == 1
        assert builtin_provider.provider_name == "target_discovery"
        assert "mcp:conflict" not in manager._loaded_provider_keys

    assert FakeMCPToolProvider.instances
    assert all(provider.closed for provider in FakeMCPToolProvider.instances)
    assert "mcp_name_conflict" in caplog.text


@pytest.mark.asyncio
async def test_manager_registers_only_read_mcp_tools_and_applies_category_filter(
    monkeypatch,
) -> None:
    import redis_sre_agent.tools.manager as manager_module

    provider_module = importlib.import_module("redis_sre_agent.tools.mcp.provider")
    read_name = "mcp_gate_a1b2c3_read"
    write_name = "mcp_gate_a1b2c3_write"
    unknown_name = "mcp_gate_a1b2c3_unknown"
    monkeypatch.setattr(
        FakeMCPToolProvider,
        "tool_specs",
        [
            (read_name, ToolCapability.DIAGNOSTICS, ToolActionKind.READ),
            (write_name, ToolCapability.DIAGNOSTICS, ToolActionKind.WRITE),
            (unknown_name, ToolCapability.UTILITIES, ToolActionKind.UNKNOWN),
        ],
    )
    monkeypatch.setattr(provider_module, "MCPToolProvider", FakeMCPToolProvider)
    monkeypatch.setattr(
        manager_module,
        "settings",
        Settings(
            _env_file=None,
            rag_enabled=False,
            mcp_servers={
                "gate": {
                    "command": sys.executable,
                    "tools": {"read_status": {"action_kind": "read"}},
                }
            },
        ),
    )

    async with ToolManager() as manager:
        names = [tool.name for tool in manager.get_tools_for_llm()]
        assert read_name in names
        assert write_name not in names
        assert unknown_name not in names
        with pytest.raises(ValueError, match="Unknown tool"):
            await manager.resolve_tool_call(write_name, {})

    async with ToolManager(
        exclude_mcp_categories=[ToolCapability.DIAGNOSTICS]
    ) as filtered_manager:
        filtered_names = [tool.name for tool in filtered_manager.get_tools_for_llm()]

    assert read_name not in filtered_names
