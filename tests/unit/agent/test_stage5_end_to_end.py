"""Stage 5 单 target 端到端 fake 测试。"""

from __future__ import annotations

import json
import sys
from typing import Any, Iterable

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from mcp import types as mcp_types
from pydantic import SecretStr

from redis_sre_agent.agent.chat_agent import ChatAgent
from redis_sre_agent.agent._compat import FakeToolCallingLLM
from redis_sre_agent.core import targets as targets_module
from redis_sre_agent.core.config import Settings
from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.targets import redis_binding
from redis_sre_agent.targets import services as target_services
from redis_sre_agent.targets.contracts import TargetHandleRecord
from redis_sre_agent.targets.registry import reset_target_integration_registry
from redis_sre_agent.tools import manager as tool_manager_module
from redis_sre_agent.tools.diagnostics.redis_command.provider import RedisCommandToolProvider
from redis_sre_agent.tools.manager import ToolManager
from redis_sre_agent.tools.mcp.provider import MCPToolProvider

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


class DynamicMCPThenTargetLLM:
    def __init__(self) -> None:
        self.tools: list[Any] = []
        self.bound_snapshots: list[set[str]] = []

    def bind_tools(self, tools: list[Any]) -> "DynamicMCPThenTargetLLM":
        self.tools = list(tools)
        self.bound_snapshots.append({tool.name for tool in self.tools})
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if not tool_messages:
            mcp_tool = next(tool for tool in self.tools if tool.name.startswith("mcp_"))
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "e2e_mcp_read",
                        "name": mcp_tool.name,
                        "args": {"detail": True},
                    }
                ],
            )
        if len(tool_messages) == 1:
            resolve_tool = next(
                tool for tool in self.tools if tool.name.endswith("resolve_redis_targets")
            )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "e2e_target_resolve",
                        "name": resolve_tool.name,
                        "args": {"query": "Prod Checkout Cache"},
                    }
                ],
            )
        if len(tool_messages) == 2:
            info_tool = next(tool for tool in self.tools if tool.name.endswith("info"))
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "e2e_redis_info",
                        "name": info_tool.name,
                        "args": {"section": "memory"},
                    }
                ],
            )
        return AIMessage(content="MCP, target, and Redis evidence collected")


class TargetDecisionRunnable:
    def __init__(self, parent: "StructuredTargetChoosingLLM") -> None:
        self.parent = parent

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self.parent.decision_calls += 1
        self.parent.decision_messages = list(messages)
        return {
            "requires_live_diagnostics": True,
            "selected_target": self.parent.selected_target,
            "reason_code": "fake_semantic_target_choice",
            "confidence": 0.91,
        }


class StructuredTargetChoosingLLM(FakeToolCallingLLM):
    def __init__(self, selected_target: str) -> None:
        super().__init__(agent_kind="chat")
        self.selected_target = selected_target
        self.decision_calls = 0
        self.decision_messages: list[Any] = []
        self.structured_methods: list[Any] = []
        self.bound_tool_snapshots: list[list[str]] = []

    def bind_tools(self, tools: list[Any]) -> FakeToolCallingLLM:
        self.bound_tool_snapshots.append(
            [str(getattr(tool, "name", "")) for tool in tools]
        )
        return super().bind_tools(tools)

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> TargetDecisionRunnable:
        if getattr(schema, "__name__", "") != "TargetSelectionDecision":
            return super().with_structured_output(schema, **_kwargs)
        self.structured_methods.append(_kwargs.get("method"))
        return TargetDecisionRunnable(self)


class FakeE2EMCPSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="external healthy")],
            structuredContent={"external_status": "healthy"},
        )


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


def make_second_instance() -> RedisInstance:
    return RedisInstance(
        id="inst-stage5-e2e-secondary",
        name="Lab Search Cache",
        connection_url=SecretStr(_URL),
        environment="lab",
        usage="search-cache",
        description="Lab fake Redis target",
        status="healthy",
    )


def make_replication_instances() -> list[RedisInstance]:
    base = make_instance()
    return [
        base.model_copy(
            update={
                "id": "redis-test-primary",
                "name": "redis-sre-primary",
                "environment": "lab",
                "usage": "replication-lab",
            }
        ),
        base.model_copy(
            update={
                "id": "redis-test-replica1",
                "name": "redis-sre-replica1",
                "environment": "lab",
                "usage": "replication-lab",
            }
        ),
        base.model_copy(
            update={
                "id": "redis-test-replica2",
                "name": "redis-sre-replica2",
                "environment": "lab",
                "usage": "replication-lab",
            }
        ),
    ]


@pytest.mark.asyncio
async def test_unscoped_live_query_uses_structured_ai_choice_for_the_only_target(
    monkeypatch,
) -> None:
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
    monkeypatch.setattr(
        tool_manager_module,
        "settings",
        Settings(_env_file=None, rag_enabled=False, mcp_servers={}),
    )

    llm = StructuredTargetChoosingLLM("Prod Checkout Cache")
    response = await ChatAgent(llm=llm).process_query(
        "Redis 有点慢，帮我看看咋回事",
        session_id="session-unscoped-single",
        user_id="user-stage5",
        context={"thread_id": "thread-unscoped-single"},
    )

    names = [envelope["name"] for envelope in response.tool_envelopes]
    assert "list_known_redis_targets" in names
    assert "resolve_redis_targets" in names
    assert "info" in names
    assert "slowlog" in names
    assert "请告诉我" not in response.response
    assert llm.decision_calls == 1
    assert llm.structured_methods == ["json_mode"]
    assert "valid JSON object" in str(llm.decision_messages[0].content)
    assert {record.public_summary.display_name for record in fake_store.records.values()} == {
        "Prod Checkout Cache"
    }


@pytest.mark.asyncio
async def test_unscoped_live_query_binds_the_target_chosen_by_ai_not_primary(
    monkeypatch,
) -> None:
    reset_target_integration_registry()
    instances = make_replication_instances()
    instances_by_id = {item.id: item for item in instances}
    fake_store = FakeTargetHandleStore()

    async def fake_get_instances():
        return instances

    async def fake_get_clusters():
        return []

    async def fake_get_instance_by_id(instance_id: str):
        return instances_by_id.get(instance_id)

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
    monkeypatch.setattr(
        tool_manager_module,
        "settings",
        Settings(_env_file=None, rag_enabled=False, mcp_servers={}),
    )

    llm = StructuredTargetChoosingLLM("redis-sre-replica2")
    response = await ChatAgent(llm=llm).process_query(
        "查询有点慢，帮我看看咋回事",
        session_id="session-unscoped-replication",
        user_id="user-stage5",
        context={"thread_id": "thread-unscoped-replication"},
    )

    names = [envelope["name"] for envelope in response.tool_envelopes]
    assert "list_known_redis_targets" in names
    assert "resolve_redis_targets" in names
    assert "info" in names
    assert "slowlog" in names
    assert "请告诉我" not in response.response
    assert llm.decision_calls == 1
    assert {record.public_summary.display_name for record in fake_store.records.values()} == {
        "redis-sre-replica2"
    }
    assert llm.bound_tool_snapshots
    assert all(
        not any(name.endswith("resolve_redis_targets") for name in snapshot)
        for snapshot in llm.bound_tool_snapshots
    )


@pytest.mark.asyncio
async def test_unscoped_live_query_uses_ai_choice_across_unrelated_targets(
    monkeypatch,
) -> None:
    reset_target_integration_registry()
    instances = [make_instance(), make_second_instance()]
    instances_by_id = {item.id: item for item in instances}
    fake_store = FakeTargetHandleStore()

    async def fake_get_instances():
        return instances

    async def fake_get_clusters():
        return []

    async def fake_get_instance_by_id(instance_id: str):
        return instances_by_id.get(instance_id)

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
    monkeypatch.setattr(
        tool_manager_module,
        "settings",
        Settings(_env_file=None, rag_enabled=False, mcp_servers={}),
    )

    llm = StructuredTargetChoosingLLM("Lab Search Cache")
    response = await ChatAgent(llm=llm).process_query(
        "Redis 有点慢，帮我看看咋回事",
        session_id="session-unscoped-multiple",
        user_id="user-stage5",
        context={"thread_id": "thread-unscoped-multiple"},
    )

    names = [envelope["name"] for envelope in response.tool_envelopes]
    assert "resolve_redis_targets" in names
    assert "list_known_redis_targets" in names
    assert "info" in names
    assert "slowlog" in names
    assert "请告诉我" not in response.response
    assert llm.decision_calls == 1
    assert {record.public_summary.display_name for record in fake_store.records.values()} == {
        "Lab Search Cache"
    }
    decision_payload = "\n".join(
        str(getattr(message, "content", "")) for message in llm.decision_messages
    )
    assert "Prod Checkout Cache" in decision_payload
    assert "Lab Search Cache" in decision_payload
    resolve_envelope = next(
        envelope
        for envelope in response.tool_envelopes
        if envelope["name"] == "resolve_redis_targets"
    )
    assert resolve_envelope["args"]["query"] == "Lab Search Cache"


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
    monkeypatch.setattr(
        tool_manager_module,
        "settings",
        Settings(_env_file=None, rag_enabled=False, mcp_servers={}),
    )

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
    assert not any(name.startswith("mcp_") for name in names)
    assert names.index("resolve_redis_targets") < names.index("info")
    assert response.search_results == []
    assert fake_store.records
    assert _PASSWORD not in payload
    assert _TOKEN not in payload
    assert _URL not in payload


@pytest.mark.asyncio
async def test_mcp_survives_target_rebinding_and_all_calls_use_tool_manager(
    monkeypatch,
) -> None:
    reset_target_integration_registry()
    instance = make_instance()
    fake_store = FakeTargetHandleStore()
    sessions: list[FakeE2EMCPSession] = []
    routed_names: list[str] = []

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

    async def fake_mcp_connect(self) -> None:
        session = FakeE2EMCPSession()
        sessions.append(session)
        self._session = session
        self._mcp_tools = [
            mcp_types.Tool(
                name="read_status",
                description="Read fake external status.",
                inputSchema={
                    "type": "object",
                    "properties": {"detail": {"type": "boolean"}},
                },
            )
        ]

    original_resolve = ToolManager.resolve_tool_call

    async def recording_resolve(self, tool_name, args, *, decision=None):
        routed_names.append(tool_name)
        return await original_resolve(self, tool_name, args, decision=decision)

    monkeypatch.setattr(targets_module, "get_instances", fake_get_instances)
    monkeypatch.setattr(targets_module, "get_clusters", fake_get_clusters)
    monkeypatch.setattr(target_services, "get_target_handle_store", lambda: fake_store)
    monkeypatch.setattr(tool_manager_module, "get_target_handle_store", lambda: fake_store)
    monkeypatch.setattr(redis_binding, "get_instance_by_id", fake_get_instance_by_id)
    monkeypatch.setattr(RedisCommandToolProvider, "get_client", fake_get_client)
    monkeypatch.setattr(MCPToolProvider, "_connect", fake_mcp_connect)
    monkeypatch.setattr(ToolManager, "resolve_tool_call", recording_resolve)
    monkeypatch.setattr(
        tool_manager_module,
        "settings",
        Settings(
            _env_file=None,
            rag_enabled=False,
            mcp_servers={
                "e2e_fake": {
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
    llm = DynamicMCPThenTargetLLM()

    response = await ChatAgent(llm=llm).process_query(
        "check external and Prod Checkout Cache memory",
        session_id="session-stage5-mcp-e2e",
        user_id="user-stage5",
        context={
            "thread_id": "thread-stage5-mcp-e2e",
            "target_query": "Prod Checkout Cache",
        },
    )

    names = [envelope["name"] for envelope in response.tool_envelopes]
    assert response.response == "MCP, target, and Redis evidence collected"
    assert names == ["read_status", "resolve_redis_targets", "info"]
    assert len(routed_names) == 3
    assert routed_names[0].startswith("mcp_")
    assert routed_names[1].endswith("resolve_redis_targets")
    assert routed_names[2].endswith("info")
    assert sessions and sessions[0].calls == [("read_status", {"detail": True})]
    assert any(
        any(name.startswith("mcp_") for name in snapshot)
        and any(name.endswith("info") for name in snapshot)
        for snapshot in llm.bound_snapshots
    )
    assert fake_store.records
