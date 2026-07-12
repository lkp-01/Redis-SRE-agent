"""真实只读 knowledge provider 与 ToolManager 动态路由测试。"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from redis_sre_agent.core import redis as redis_core
from redis_sre_agent.core.config import Settings
from redis_sre_agent.tools.knowledge.knowledge_base import KnowledgeBaseToolProvider
from redis_sre_agent.tools.manager import ToolManager
from redis_sre_agent.tools.models import ToolActionKind, ToolCapability


def _search_result() -> dict[str, Any]:
    return {
        "status": "success",
        "query": "latency",
        "retrieval_kind": "knowledge_search",
        "retrieval_label": "Knowledge search",
        "results_count": 1,
        "results": [
            {
                "title": "Latency runbook",
                "source": "file://shared/latency.md",
                "document_hash": "doc",
                "chunk_index": 0,
                "score": 0.1,
                "content": "Inspect SLOWLOG.",
            }
        ],
    }


def test_provider_exposes_exactly_one_read_search_tool() -> None:
    provider = KnowledgeBaseToolProvider()
    schemas = provider.create_tool_schemas()
    tools = provider.tools()

    assert provider.provider_name == "knowledge"
    assert provider.requires_redis_instance is False
    assert len(schemas) == 1
    assert schemas[0].name.endswith("search")
    assert schemas[0].capability is ToolCapability.KNOWLEDGE
    assert schemas[0].parameters["required"] == ["query"]
    assert tools[0].metadata.action_kind is ToolActionKind.READ
    assert not hasattr(provider, "ingest")


@pytest.mark.asyncio
async def test_provider_search_uses_shared_helper(monkeypatch) -> None:
    import redis_sre_agent.tools.knowledge.knowledge_base as provider_module

    captured: dict[str, Any] = {}

    async def fake_helper(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _search_result()

    monkeypatch.setattr(provider_module, "search_knowledge_base_helper", fake_helper)
    provider = KnowledgeBaseToolProvider()

    result = await provider.search(
        query="latency",
        limit=4,
        offset=2,
        version="latest",
        distance_threshold=0.7,
    )

    assert result == _search_result()
    assert captured == {
        "query": "latency",
        "limit": 4,
        "offset": 2,
        "version": "latest",
        "distance_threshold": 0.7,
    }


@pytest.mark.asyncio
async def test_tool_manager_ready_routes_dynamic_knowledge_tool(monkeypatch) -> None:
    import redis_sre_agent.tools.manager as manager_module
    import redis_sre_agent.tools.knowledge.knowledge_base as provider_module

    config = Settings(
        _env_file=None,
        rag_enabled=True,
        embedding_api_key=SecretStr("TEST_EMBEDDING_KEY"),
    )
    monkeypatch.setattr(manager_module, "settings", config)

    async def ready(_config=None):
        return redis_core.RAGReadiness("ready", "ready", "RAG 已就绪。")

    async def fake_helper(**_kwargs: Any) -> dict[str, Any]:
        return _search_result()

    monkeypatch.setattr(redis_core, "get_rag_readiness", ready)
    monkeypatch.setattr(provider_module, "search_knowledge_base_helper", fake_helper)

    async with ToolManager() as manager:
        name = next(
            tool.name
            for tool in manager.get_tools_for_llm()
            if tool.name.startswith("knowledge_") and tool.name.endswith("search")
        )
        result = await manager.resolve_tool_call(name, {"query": "latency"})

    assert result == _search_result()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "state", "reason_code"),
    [
        (False, "disabled", "disabled"),
        (True, "not_ready", "schema_mismatch"),
    ],
)
async def test_tool_manager_unavailable_states_never_expose_search(
    monkeypatch,
    enabled: bool,
    state: str,
    reason_code: str,
) -> None:
    import redis_sre_agent.tools.manager as manager_module

    monkeypatch.setattr(
        manager_module,
        "settings",
        Settings(
            _env_file=None,
            rag_enabled=enabled,
            embedding_api_key=SecretStr("TEST_EMBEDDING_KEY") if enabled else None,
        ),
    )

    async def readiness(_config=None):
        return redis_core.RAGReadiness(state, reason_code, "安全原因。")

    monkeypatch.setattr(redis_core, "get_rag_readiness", readiness)

    async with ToolManager() as manager:
        names = [tool.name for tool in manager.get_tools_for_llm()]

    assert not any(name.startswith("knowledge_") for name in names)
