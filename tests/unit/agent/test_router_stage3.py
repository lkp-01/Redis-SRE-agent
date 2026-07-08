"""阶段三 router 到 ToolManager 的曳光弹测试。"""

from __future__ import annotations

import json
from typing import Iterable

import pytest
from pydantic import SecretStr

from redis_sre_agent.agent.router import AgentType, route_to_appropriate_agent
from redis_sre_agent.core import targets as targets_module
from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.targets import redis_binding
from redis_sre_agent.targets import services as target_services
from redis_sre_agent.targets.contracts import TargetHandleRecord
from redis_sre_agent.targets.registry import reset_target_integration_registry
from redis_sre_agent.tools import manager as tool_manager_module
from redis_sre_agent.tools.manager import ToolManager


class FakeTargetHandleStore:
    """内存版 handle store，保证测试不访问真实 Redis。"""

    def __init__(self) -> None:
        self.records: dict[str, TargetHandleRecord] = {}

    async def save_records(self, records: Iterable[TargetHandleRecord]) -> None:
        for record in records:
            self.records[record.target_handle] = record

    async def get_record(self, target_handle: str) -> TargetHandleRecord | None:
        return self.records.get(target_handle)

    async def get_records(self, target_handles: Iterable[str]) -> dict[str, TargetHandleRecord]:
        return {
            handle: self.records[handle]
            for handle in target_handles
            if handle in self.records
        }


def make_instance() -> RedisInstance:
    return RedisInstance(
        id="inst-prod-checkout-cache",
        name="Prod Checkout Cache",
        connection_url=SecretStr("FAKE_TEST_REDIS_CONNECTION_REF"),
        environment="production",
        usage="cache",
        description="Checkout service cache",
        monitoring_identifier="checkout-cache-prod",
        status="healthy",
        extension_data={"aliases": ["checkout prod"]},
    )


@pytest.mark.asyncio
async def test_router_uses_original_agent_type_names() -> None:
    assert await route_to_appropriate_agent("check redis") is AgentType.REDIS_CHAT
    assert await route_to_appropriate_agent("deep triage redis") is AgentType.REDIS_TRIAGE


@pytest.mark.asyncio
async def test_router_toolmanager_chain_resolves_target_and_calls_info(monkeypatch) -> None:
    reset_target_integration_registry()
    instance = make_instance()
    fake_store = FakeTargetHandleStore()

    async def fake_get_instances():
        return [instance]

    async def fake_get_clusters():
        return []

    async def fake_get_instance_by_id(instance_id: str):
        return instance if instance_id == instance.id else None

    monkeypatch.setattr(targets_module, "get_instances", fake_get_instances)
    monkeypatch.setattr(targets_module, "get_clusters", fake_get_clusters)
    monkeypatch.setattr(target_services, "get_target_handle_store", lambda: fake_store)
    monkeypatch.setattr(tool_manager_module, "get_target_handle_store", lambda: fake_store)
    monkeypatch.setattr(redis_binding, "get_instance_by_id", fake_get_instance_by_id)

    route = await route_to_appropriate_agent(instance.name)
    async with ToolManager() as manager:
        discovery_tools = manager.get_tools_by_provider_names(["target_discovery"])
        resolve_tool = next(tool.name for tool in discovery_tools if tool.name.endswith("resolve_redis_targets"))
        target_result = await manager.resolve_tool_call(
            resolve_tool,
            {
                "query": instance.name,
                "attach_tools": True,
                "preferred_capabilities": ["diagnostics"],
            },
        )
        redis_tools = manager.get_tools_by_provider_names(["redis_command"])
        info_tool = next(tool.name for tool in redis_tools if tool.name.endswith("info"))
        info_result = await manager.resolve_tool_call(info_tool, {})

    payload = json.dumps({"target": target_result, "info": info_result}, ensure_ascii=False)
    assert route is AgentType.REDIS_CHAT
    assert target_result["status"] == "resolved"
    assert info_result["status"] == "success"
    assert info_result["tool"] == "info"
    assert info_result["target_handle"].startswith("tgt_")
    assert fake_store.records
    assert "FAKE_TEST_REDIS_CONNECTION_REF" not in payload
