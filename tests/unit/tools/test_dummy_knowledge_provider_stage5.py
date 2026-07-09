"""Stage 5 dummy knowledge provider 测试。"""

from __future__ import annotations

import pytest

from redis_sre_agent.tools.knowledge.knowledge_base import KnowledgeBaseToolProvider
from redis_sre_agent.tools.manager import ToolManager
from redis_sre_agent.tools.models import ToolCapability


def test_dummy_knowledge_provider_schema_shape() -> None:
    provider = KnowledgeBaseToolProvider()
    schemas = provider.create_tool_schemas()

    assert provider.provider_name == "knowledge"
    assert provider.requires_redis_instance is False
    assert len(schemas) == 1
    assert schemas[0].name.endswith("search")
    assert schemas[0].capability is ToolCapability.KNOWLEDGE
    assert schemas[0].parameters["required"] == ["query"]


@pytest.mark.asyncio
async def test_dummy_knowledge_search_returns_empty_stage5_slot() -> None:
    provider = KnowledgeBaseToolProvider()

    result = await provider.search(query="memory")

    assert result["status"] == "success"
    assert result["results"] == []
    assert result["retrieval_kind"] == "dummy_knowledge"
    assert "Stage 5" in result["message"]


@pytest.mark.asyncio
async def test_tool_manager_loads_target_discovery_and_dummy_knowledge() -> None:
    async with ToolManager() as manager:
        tools = manager.get_tools()
        names = [tool.name for tool in tools]

    assert any(name.endswith("resolve_redis_targets") for name in names)
    assert any(name.startswith("knowledge_") and name.endswith("search") for name in names)
