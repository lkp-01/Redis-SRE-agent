"""Stage 5 单 target 端到端 fake 测试。"""

from __future__ import annotations

import json
from typing import Any, Iterable

import pytest
from pydantic import SecretStr

from redis_sre_agent.agent.chat_agent import ChatAgent
from redis_sre_agent.agent._compat import FakeToolCallingLLM
from redis_sre_agent.core import targets as targets_module
from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.targets import redis_binding
from redis_sre_agent.targets import services as target_services
from redis_sre_agent.targets.contracts import TargetHandleRecord
from redis_sre_agent.targets.registry import reset_target_integration_registry
from redis_sre_agent.tools import manager as tool_manager_module
from redis_sre_agent.tools.diagnostics.redis_command.provider import RedisCommandToolProvider

_PASSWORD = "stage5-e2e-password"
_TOKEN = "stage5-e2e-token"
_URL = f"redis://default:{_PASSWORD}@cache.internal:6379/0"


class FakeTargetHandleStore:
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


class FakeE2ERedisClient:
    async def info(self, section: str | None = None) -> dict[str, Any]:
        if section == "memory":
            return {"used_memory": 4096, "used_memory_human": "4K", "maxmemory": 8192}
        if section == "stats":
            return {"instantaneous_ops_per_sec": 11, "evicted_keys": 0}
        if section == "replication":
            return {"role": "master", "connected_slaves": 1}
        return {"redis_version": "7.2.0"}

    async def client_list(self, _type: str | None = None) -> list[dict[str, Any]]:
        return [{"id": "1", "cmd": "get"}, {"id": "2", "cmd": "set"}]

    async def slowlog_get(self, count: int) -> list[dict[str, Any]]:
        return [
            {
                "id": 7,
                "start_time": 1_700_000_000,
                "duration": 1000,
                "command": ["SET", "api_token", _TOKEN],
                "client_address": "10.0.0.2:5000",
                "client_name": "worker",
            }
        ][:count]

    async def memory_stats(self) -> dict[str, Any]:
        return {"peak.allocated": 8192, "clients.normal": 128}

    async def config_get(self, pattern: str) -> dict[str, str]:
        return {"maxmemory": "8192", "requirepass": _PASSWORD}

    async def execute_command(self, command: str, *args: Any) -> Any:
        if command == "ROLE":
            return ["master", 0, []]
        raise RuntimeError(f"unsupported command: {command}")

    async def aclose(self) -> None:
        return None


def make_instance() -> RedisInstance:
    return RedisInstance(
        id="inst-stage5-e2e",
        name="Prod Checkout Cache",
        connection_url=SecretStr(_URL),
        environment="production",
        usage="cache",
        description="Checkout Redis cache",
        monitoring_identifier="checkout-cache-prod",
        extension_data={"aliases": ["checkout prod"]},
        status="healthy",
    )


@pytest.mark.asyncio
async def test_stage5_agent_resolves_target_binds_tools_and_reports(monkeypatch) -> None:
    reset_target_integration_registry()
    instance = make_instance()
    fake_store = FakeTargetHandleStore()

    async def fake_get_instances():
        return [instance]

    async def fake_get_clusters():
        return []

    async def fake_get_instance_by_id(instance_id: str):
        return instance if instance_id == instance.id else None

    def fake_get_client(self):
        if self._client is None:
            self._client = FakeE2ERedisClient()
        return self._client

    monkeypatch.setattr(targets_module, "get_instances", fake_get_instances)
    monkeypatch.setattr(targets_module, "get_clusters", fake_get_clusters)
    monkeypatch.setattr(target_services, "get_target_handle_store", lambda: fake_store)
    monkeypatch.setattr(tool_manager_module, "get_target_handle_store", lambda: fake_store)
    monkeypatch.setattr(redis_binding, "get_instance_by_id", fake_get_instance_by_id)
    monkeypatch.setattr(RedisCommandToolProvider, "get_client", fake_get_client)

    response = await ChatAgent(llm=FakeToolCallingLLM(agent_kind="chat")).process_query(
        "check Prod Checkout Cache memory and slowlog",
        session_id="session-stage5-e2e",
        user_id="user-stage5",
        context={"thread_id": "thread-stage5-e2e", "target_query": "Prod Checkout Cache"},
    )

    names = [envelope["name"] for envelope in response.tool_envelopes]
    payload = json.dumps(response.model_dump(mode="json"), ensure_ascii=False, default=str)
    assert response.response
    assert "resolve_redis_targets" in names
    assert "info" in names
    assert "memory_stats" in names
    assert "client_list" in names
    assert "slowlog" in names
    assert names.index("resolve_redis_targets") < names.index("info")
    assert response.search_results == []
    assert fake_store.records
    assert _PASSWORD not in payload
    assert _TOKEN not in payload
    assert _URL not in payload
