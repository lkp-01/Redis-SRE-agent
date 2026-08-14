"""阶段三 ToolManager 与 dummy provider 测试。"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from redis_sre_agent.core import redis as redis_core
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


_EXPECTED_REDIS_OPERATIONS = {
    "info",
    "slowlog",
    "acl_log",
    "config_get",
    "client_list",
    "cluster_info",
    "replication_info",
    "memory_stats",
    "bigkey_scan",
    "sample_keys",
    "search_indexes",
    "search_index_info",
}


class FakeInfoClient:
    """只覆盖 INFO 的 fake client，避免链路测试访问真实 Redis。"""

    async def info(self, section=None):
        return {"redis_version": "stage4-fake", "role": "master", "section": section or "all"}

    async def aclose(self):
        return None


def make_instance() -> RedisInstance:
    return RedisInstance(
        id="inst-local-cache",
        name="Local Cache",
        connection_url=SecretStr("redis://localhost:6379/0"),
        environment="test",
        usage="cache",
        description="Local test cache",
    )


@pytest.mark.asyncio
async def test_tool_manager_loads_target_discovery_and_redis_info_tool(monkeypatch) -> None:
    def fake_get_client(self):
        if self._client is None:
            self._client = FakeInfoClient()
        return self._client

    monkeypatch.setattr(RedisCommandToolProvider, "get_client", fake_get_client)

    async with ToolManager(redis_instance=make_instance()) as manager:
        tools = manager.get_tools()
        names = [tool.name for tool in tools]

        assert any(name.endswith("list_known_redis_targets") for name in names)
        assert any(name.endswith("resolve_redis_targets") for name in names)
        assert any(name.endswith("info") for name in names)

        info_tool = next(name for name in names if name.endswith("info"))
        result = await manager.resolve_tool_call(info_tool, {})

    assert result["status"] == "success"
    assert result["section"] == "all"
    assert result["data"]["redis_version"] == "stage4-fake"


@pytest.mark.asyncio
async def test_tool_manager_registers_exact_redis_diagnostic_contract() -> None:
    async with ToolManager(redis_instance=make_instance()) as manager:
        providers = manager.get_providers_for_capability(ToolCapability.DIAGNOSTICS)
        provider = next(item for item in providers if item.provider_name == "redis_command")
        definitions = manager.get_tools_by_provider_names(["redis_command"])
        operations = {provider.resolve_operation(tool.name, {}) for tool in definitions}

    assert len(definitions) == len(_EXPECTED_REDIS_OPERATIONS)
    assert operations == _EXPECTED_REDIS_OPERATIONS


@pytest.mark.asyncio
async def test_tool_manager_disabled_rag_never_checks_or_loads_knowledge(monkeypatch) -> None:
    import redis_sre_agent.tools.manager as manager_module

    monkeypatch.setattr(
        manager_module,
        "settings",
        Settings(_env_file=None, rag_enabled=False),
    )

    async def forbidden_readiness(*_args, **_kwargs):
        raise AssertionError("disabled RAG must not perform readiness checks")

    monkeypatch.setattr(redis_core, "get_rag_readiness", forbidden_readiness)

    async with ToolManager() as manager:
        names = [tool.name for tool in manager.get_tools_for_llm()]
        readiness = manager.rag_readiness

    assert any(name.endswith("resolve_redis_targets") for name in names)
    assert not any(name.startswith("knowledge_") for name in names)
    assert readiness.state == "disabled"


@pytest.mark.asyncio
async def test_tool_manager_enabled_not_ready_hides_knowledge_tool(monkeypatch) -> None:
    import redis_sre_agent.tools.manager as manager_module

    config = Settings(
        _env_file=None,
        rag_enabled=True,
        embedding_api_key=SecretStr("TEST_EMBEDDING_KEY"),
    )
    monkeypatch.setattr(manager_module, "settings", config)

    async def fake_readiness(_config=None):
        return redis_core.RAGReadiness(
            state="not_ready",
            reason_code="index_missing",
            message="knowledge index is missing",
        )

    monkeypatch.setattr(redis_core, "get_rag_readiness", fake_readiness)

    async with ToolManager() as manager:
        names = [tool.name for tool in manager.get_tools_for_llm()]
        readiness = manager.rag_readiness

    assert not any(name.startswith("knowledge_") for name in names)
    assert readiness.reason_code == "index_missing"


@pytest.mark.asyncio
async def test_tool_manager_loads_knowledge_only_when_ready(monkeypatch) -> None:
    import redis_sre_agent.tools.manager as manager_module

    config = Settings(
        _env_file=None,
        rag_enabled=True,
        embedding_api_key=SecretStr("TEST_EMBEDDING_KEY"),
    )
    monkeypatch.setattr(manager_module, "settings", config)

    async def fake_readiness(_config=None):
        return redis_core.RAGReadiness(
            state="ready",
            reason_code="ready",
            message="RAG is ready",
        )

    monkeypatch.setattr(redis_core, "get_rag_readiness", fake_readiness)

    async with ToolManager() as manager:
        names = [tool.name for tool in manager.get_tools_for_llm()]

    assert any(name.startswith("knowledge_") and name.endswith("search") for name in names)


@pytest.mark.asyncio
async def test_tool_manager_execute_tool_calls_redacts_fallback_errors(monkeypatch) -> None:
    secret = "manager-secret-value"
    raw_url = f"redis://default:{secret}@cache.internal:6379/0"

    async with ToolManager() as manager:
        async def fake_resolve_tool_call(tool_name, args):
            raise RuntimeError(f"failed password={secret} token={secret} url={raw_url}")

        monkeypatch.setattr(manager, "resolve_tool_call", fake_resolve_tool_call)
        result = await manager.execute_tool_calls([{"name": "fake_tool", "args": {}}])

    payload = str(result)
    assert result[0]["status"] == "failed"
    assert secret not in payload
    assert raw_url not in payload
    assert "[REDACTED]" in payload


def test_llm_tool_limit_drops_mcp_before_any_builtin_tool() -> None:
    manager = ToolManager()

    def make_tool(index: int, *, provider_name: str, capability: ToolCapability) -> Tool:
        name = f"{provider_name}_{index:02d}"

        async def invoke(_args):
            return {"status": "success"}

        definition = ToolDefinition(
            name=name,
            description="Test tool.",
            capability=capability,
            parameters={"type": "object", "properties": {}},
        )
        return Tool(
            metadata=ToolMetadata(
                name=name,
                description=definition.description,
                capability=capability,
                provider_name=provider_name,
                action_kind=ToolActionKind.READ,
            ),
            definition=definition,
            invoke=invoke,
        )

    manager._tools = [
        make_tool(index, provider_name="builtin_custom", capability=ToolCapability.ADMIN)
        for index in range(64)
    ]
    manager._tools.append(
        make_tool(99, provider_name="mcp_external", capability=ToolCapability.DIAGNOSTICS)
    )

    selected = manager.get_tools_for_llm(max_tools=64)
    selected_names = {tool.name for tool in selected}

    assert len(selected) == 64
    assert "mcp_external_99" not in selected_names
    assert {f"builtin_custom_{index:02d}" for index in range(64)} == selected_names
