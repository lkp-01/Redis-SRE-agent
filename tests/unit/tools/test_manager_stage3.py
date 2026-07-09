"""阶段三 ToolManager 与 dummy provider 测试。"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.tools.diagnostics.redis_command.provider import RedisCommandToolProvider
from redis_sre_agent.tools.manager import ToolManager


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
