"""Stage 5 ChatAgent 主链路测试。"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import SecretStr
from langgraph.graph import StateGraph

from redis_sre_agent.agent import chat_agent as chat_agent_module
from redis_sre_agent.agent._compat import FakeToolCallingLLM
from redis_sre_agent.agent.chat_agent import ChatAgent, get_chat_agent
from redis_sre_agent.agent.models import AgentResponse
from redis_sre_agent.core import llm_helpers
from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.tools.diagnostics.redis_command.provider import RedisCommandToolProvider
from redis_sre_agent.tools.manager import ToolManager

_PASSWORD = "stage5-chat-password"
_TOKEN = "stage5-chat-token"
_URL = f"redis://default:{_PASSWORD}@cache.internal:6379/0"


class FakeChatRedisClient:
    async def info(self, section: str | None = None) -> dict[str, Any]:
        if section == "memory":
            return {"used_memory": 2048, "used_memory_human": "2K", "maxmemory": 4096}
        if section == "stats":
            return {"instantaneous_ops_per_sec": 42, "evicted_keys": 0}
        if section == "replication":
            return {"role": "master", "connected_slaves": 0}
        return {"redis_version": "7.2.0"}

    async def client_list(self, _type: str | None = None) -> list[dict[str, Any]]:
        return [{"id": "1", "addr": "10.0.0.2:5000", "cmd": "get"}]

    async def slowlog_get(self, count: int) -> list[dict[str, Any]]:
        return [
            {
                "id": 1,
                "start_time": 1_700_000_000,
                "duration": 900,
                "command": ["SET", "api_token", _TOKEN],
                "client_address": "10.0.0.2:5000",
                "client_name": "worker",
            }
        ][:count]

    async def memory_stats(self) -> dict[str, Any]:
        return {"peak.allocated": 4096, "clients.normal": 64}

    async def config_get(self, pattern: str) -> dict[str, Any]:
        return {"maxmemory": "4096", "requirepass": _PASSWORD}

    async def execute_command(self, command: str, *args: Any) -> Any:
        if command == "ROLE":
            return ["master", 0, []]
        raise RuntimeError(f"unsupported command: {command}")

    async def aclose(self) -> None:
        return None


def make_instance() -> RedisInstance:
    return RedisInstance(
        id="inst-stage5-chat",
        name="Stage 5 Chat Redis",
        connection_url=SecretStr(_URL),
        environment="test",
        usage="cache",
        description="Stage 5 fake chat target",
    )


def _assert_no_sensitive_payload(value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    assert _PASSWORD not in payload
    assert _TOKEN not in payload
    assert _URL not in payload
    assert "SecretStr" not in payload


def test_chat_agent_prefers_explicit_llm() -> None:
    explicit_llm = object()

    agent = ChatAgent(llm=explicit_llm)

    assert agent.llm is explicit_llm


def test_chat_agent_uses_real_factory_only_when_key_is_configured(monkeypatch) -> None:
    real_llm = object()
    monkeypatch.setattr(chat_agent_module.settings, "openai_api_key", SecretStr("configured"))
    monkeypatch.setattr(llm_helpers, "create_llm", lambda: real_llm)

    assert ChatAgent().llm is real_llm

    monkeypatch.setattr(chat_agent_module.settings, "openai_api_key", None)
    assert isinstance(ChatAgent().llm, FakeToolCallingLLM)


@pytest.mark.asyncio
async def test_get_chat_agent_returns_chat_agent() -> None:
    agent = get_chat_agent(redis_instance=make_instance())

    assert isinstance(agent, ChatAgent)


@pytest.mark.asyncio
async def test_chat_agent_builds_real_stategraph() -> None:
    async with ToolManager() as manager:
        workflow = ChatAgent()._build_workflow(manager)
        app = workflow.compile()

    assert isinstance(workflow, StateGraph)
    assert hasattr(app, "ainvoke")


@pytest.mark.asyncio
async def test_chat_agent_process_query_collects_redis_evidence(monkeypatch) -> None:
    def fake_get_client(self):
        if self._client is None:
            self._client = FakeChatRedisClient()
        return self._client

    monkeypatch.setattr(RedisCommandToolProvider, "get_client", fake_get_client)
    agent = ChatAgent(redis_instance=make_instance())

    response = await agent.process_query(
        "check redis memory and clients",
        session_id="session-stage5-chat",
        user_id="user-stage5",
        context={"thread_id": "thread-stage5-chat"},
    )

    names = [envelope["name"] for envelope in response.tool_envelopes]
    assert isinstance(response, AgentResponse)
    assert "info" in names
    assert "memory_stats" in names
    assert "client_list" in names
    assert "slowlog" in names
    assert "Redis 诊断摘要" in response.response
    assert "Evidence 摘要" in response.response
    assert "下一步建议" in response.response
    _assert_no_sensitive_payload(response.model_dump(mode="json"))
