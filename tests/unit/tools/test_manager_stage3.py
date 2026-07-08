"""阶段三 ToolManager 与 dummy provider 测试。"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.tools.manager import ToolManager


def make_instance() -> RedisInstance:
    return RedisInstance(
        id="inst-local-cache",
        name="Local Cache",
        connection_url=SecretStr("FAKE_TEST_REDIS_CONNECTION_REF"),
        environment="test",
        usage="cache",
        description="Local test cache",
    )


@pytest.mark.asyncio
async def test_tool_manager_loads_target_discovery_and_dummy_redis_tool() -> None:
    async with ToolManager(redis_instance=make_instance()) as manager:
        tools = manager.get_tools()
        names = [tool.name for tool in tools]

        assert any(name.endswith("list_known_redis_targets") for name in names)
        assert any(name.endswith("resolve_redis_targets") for name in names)
        assert any(name.endswith("info") for name in names)

        info_tool = next(name for name in names if name.endswith("info"))
        result = await manager.resolve_tool_call(info_tool, {})

    assert result["status"] == "success"
    assert result["tool"] == "info"
    assert result["mode"] == "mock"
    assert result["target_handle"] == "inst-local-cache"
    assert result["target_name"] == "Local Cache"
