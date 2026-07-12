"""Stage 5 SRELangGraphAgent facade 测试。"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from pydantic import SecretStr
from langgraph.graph import StateGraph

from redis_sre_agent.agent import langgraph_agent as langgraph_agent_module
from redis_sre_agent.agent._compat import FakeToolCallingLLM
from redis_sre_agent.agent.langgraph_agent import SRELangGraphAgent, get_sre_agent
from redis_sre_agent.agent.models import (
    AgentResponse,
    Recommendation,
    RecommendationStep,
    Topic,
    TopicsList,
)
from redis_sre_agent.core import llm_helpers
from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.tools.diagnostics.redis_command.provider import RedisCommandToolProvider
from redis_sre_agent.tools.manager import ToolManager

_PASSWORD = "stage5-triage-password"
_URL = f"redis://default:{_PASSWORD}@cache.internal:6379/0"


class _StructuredPipelineLLM:
    def __init__(self, parent: "TriagePipelineLLM", schema: type[Any]) -> None:
        self.parent = parent
        self.schema = schema

    async def ainvoke(self, messages: list[Any]) -> Any:
        if self.schema is TopicsList:
            self.parent.stages.append("topics")
            return TopicsList(
                items=[
                    Topic(
                        id="T1",
                        title="Memory pressure",
                        category="Performance",
                        severity="high",
                        evidence_keys=["redis_command_info"],
                    )
                ]
            )
        if self.schema is Recommendation:
            self.parent.stages.append("recommendation")
            return Recommendation(
                topic_id="T1",
                title="Memory pressure",
                steps=[RecommendationStep(description="Inspect the evidence-backed metric")],
            )
        raise AssertionError(f"unexpected structured schema: {self.schema}")


class TriagePipelineLLM:
    def __init__(self, *, composer_error: bool = False) -> None:
        self.tools: list[Any] = []
        self.stages: list[str] = []
        self.structured_methods: list[str | None] = []
        self.composer_error = composer_error

    def bind_tools(self, tools: list[Any]) -> "TriagePipelineLLM":
        self.tools = list(tools)
        return self

    def with_structured_output(
        self,
        schema: type[Any],
        **kwargs: Any,
    ) -> _StructuredPipelineLLM:
        self.structured_methods.append(kwargs.get("method"))
        return _StructuredPipelineLLM(self, schema)

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        system_text = "\n".join(
            str(message.content) for message in messages if isinstance(message, SystemMessage)
        )
        if "careful technical editor" in system_text:
            self.stages.append("composer")
            if self.composer_error:
                raise RuntimeError("composer failed")
            return AIMessage(
                content=(
                    "## Initial Assessment\nLLM assessment\n\n"
                    "## What I'm Seeing\nEvidence-backed finding\n\n"
                    "## My Recommendation\nLLM recommendation\n\n"
                    "## Supporting Info\ninfo"
                )
            )
        if "research and then synthesize recommendations" in system_text:
            return AIMessage(content="Evidence is sufficient", tool_calls=[])
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="Initial evidence assessment")
        info_tool = next(tool.name for tool in self.tools if tool.name.endswith("info"))
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_triage_pipeline",
                    "name": info_tool,
                    "args": {"section": "memory"},
                }
            ],
        )


class FakeTriageRedisClient:
    async def info(self, section: str | None = None) -> dict[str, Any]:
        if section == "memory":
            return {"used_memory_human": "4K", "used_memory": 4096}
        if section == "stats":
            return {"instantaneous_ops_per_sec": 7}
        if section == "clients":
            return {"connected_clients": 2}
        if section == "keyspace":
            return {"db0": {"keys": 12}}
        if section == "replication":
            return {"role": "replica", "master_host": "10.0.0.1"}
        return {"redis_version": "7.2.0"}

    async def client_list(self, _type: str | None = None) -> list[dict[str, Any]]:
        return [{"id": "1"}, {"id": "2"}]

    async def slowlog_get(self, count: int) -> list[dict[str, Any]]:
        return []

    async def memory_stats(self) -> dict[str, Any]:
        return {"peak.allocated": 8192}

    async def config_get(self, pattern: str) -> dict[str, str]:
        return {"maxmemory": "8192"}

    async def acl_log(self, count: int) -> list[dict[str, Any]]:
        return [{"count": 1, "reason": "command", "context": "multi"}]

    async def cluster(self, subcommand: str) -> Any:
        if subcommand == "INFO":
            raise RuntimeError("unknown command 'CLUSTER'")
        return ""

    async def execute_command(self, command: str, *args: Any) -> Any:
        if command == "ROLE":
            return ["replica", "10.0.0.1", 6379, "connected", 10]
        if command == "FT._LIST":
            return ["idx:orders"]
        if command == "FT.INFO":
            return ["index_name", args[0], "num_docs", "3"]
        raise RuntimeError(f"unsupported command: {command}")

    async def aclose(self) -> None:
        return None


def make_instance() -> RedisInstance:
    return RedisInstance(
        id="inst-stage5-triage",
        name="Stage 5 Triage Redis",
        connection_url=SecretStr(_URL),
        environment="test",
        usage="cache",
        description="Stage 5 fake triage target",
    )


def test_langgraph_agent_prefers_explicit_llm() -> None:
    explicit_llm = object()

    agent = SRELangGraphAgent(llm=explicit_llm)

    assert agent.llm is explicit_llm


def test_langgraph_agent_uses_real_factory_only_when_key_is_configured(monkeypatch) -> None:
    real_llm = object()
    monkeypatch.setattr(langgraph_agent_module.settings, "openai_api_key", SecretStr("configured"))
    monkeypatch.setattr(llm_helpers, "create_llm", lambda: real_llm)

    assert SRELangGraphAgent().llm is real_llm

    monkeypatch.setattr(langgraph_agent_module.settings, "openai_api_key", None)
    assert isinstance(SRELangGraphAgent().llm, FakeToolCallingLLM)


def test_get_sre_agent_returns_new_instance_each_time() -> None:
    first = get_sre_agent(redis_instance=make_instance())
    second = get_sre_agent(redis_instance=make_instance())

    assert isinstance(first, SRELangGraphAgent)
    assert isinstance(second, SRELangGraphAgent)
    assert first is not second


@pytest.mark.asyncio
async def test_langgraph_agent_builds_real_stategraph() -> None:
    async with ToolManager() as manager:
        agent = SRELangGraphAgent()
        workflow = agent._build_workflow(manager)
        app = workflow.compile()

    assert isinstance(workflow, StateGraph)
    assert hasattr(app, "ainvoke")


@pytest.mark.asyncio
async def test_langgraph_agent_triage_facade_collects_extended_evidence(monkeypatch) -> None:
    def fake_get_client(self):
        if self._client is None:
            self._client = FakeTriageRedisClient()
        return self._client

    monkeypatch.setattr(RedisCommandToolProvider, "get_client", fake_get_client)
    agent = SRELangGraphAgent(
        redis_instance=make_instance(),
        llm=FakeToolCallingLLM(agent_kind="triage"),
    )

    response = await agent.process_query(
        "comprehensive triage redis",
        session_id="session-stage5-triage",
        user_id="user-stage5",
        context={"thread_id": "thread-stage5-triage"},
    )

    names = [envelope["name"] for envelope in response.tool_envelopes]
    payload = json.dumps(response.model_dump(mode="json"), ensure_ascii=False, default=str)
    assert isinstance(response, AgentResponse)
    assert "## Initial Assessment" in response.response
    assert "cluster_info" in names
    assert "search_indexes" in names
    assert "search_index_info" in names
    assert any(envelope["name"] == "cluster_info" and envelope["status"] == "error" for envelope in response.tool_envelopes)
    assert _PASSWORD not in payload
    assert _URL not in payload
    assert any(name.startswith("langgraph") for name in sys.modules)
    assert isinstance(agent.llm, FakeToolCallingLLM)


@pytest.mark.asyncio
async def test_langgraph_agent_resume_is_future_slot() -> None:
    response = await SRELangGraphAgent().resume_query(session_id="s", user_id=None)

    assert "future-stage slot" in response.response
    assert response.tool_envelopes == []


@pytest.mark.asyncio
async def test_triage_runs_topics_recommendation_and_composer(monkeypatch) -> None:
    def fake_get_client(self):
        if self._client is None:
            self._client = FakeTriageRedisClient()
        return self._client

    monkeypatch.setattr(RedisCommandToolProvider, "get_client", fake_get_client)
    llm = TriagePipelineLLM()
    agent = SRELangGraphAgent(redis_instance=make_instance(), llm=llm)

    response = await agent.process_query(
        "triage memory",
        session_id="session-triage-pipeline",
        user_id=None,
        context={"thread_id": "thread-triage-pipeline"},
    )

    assert response.response.startswith("## Initial Assessment")
    assert llm.stages == ["topics", "recommendation", "composer"]
    assert llm.structured_methods == ["function_calling", "function_calling"]
    assert "Redis 深度诊断报告" not in response.response


@pytest.mark.asyncio
async def test_triage_composer_failure_uses_deterministic_fallback(monkeypatch) -> None:
    def fake_get_client(self):
        if self._client is None:
            self._client = FakeTriageRedisClient()
        return self._client

    monkeypatch.setattr(RedisCommandToolProvider, "get_client", fake_get_client)
    llm = TriagePipelineLLM(composer_error=True)
    agent = SRELangGraphAgent(redis_instance=make_instance(), llm=llm)

    response = await agent.process_query(
        "triage memory",
        session_id="session-composer-failure",
        user_id=None,
        context={"thread_id": "thread-composer-failure"},
    )

    assert llm.stages == ["topics", "recommendation", "composer"]
    assert "Redis 深度诊断报告" in response.response
