"""Stage 5 ChatAgent 主链路测试。"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from mcp import types as mcp_types
from pydantic import SecretStr
from langgraph.graph import StateGraph

from redis_sre_agent.agent import chat_agent as chat_agent_module
from redis_sre_agent.agent._compat import FakeToolCallingLLM
from redis_sre_agent.agent.chat_agent import CHAT_SYSTEM_PROMPT, ChatAgent, get_chat_agent
from redis_sre_agent.agent.models import AgentResponse
from redis_sre_agent.core import llm_helpers
from redis_sre_agent.core import redis as redis_core
from redis_sre_agent.core.config import Settings
from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.tools.diagnostics.redis_command.provider import RedisCommandToolProvider
from redis_sre_agent.tools.manager import ToolManager
from redis_sre_agent.tools.mcp.provider import MCPToolProvider

_PASSWORD = "stage5-chat-password"
_TOKEN = "stage5-chat-token"
_URL = f"redis://default:{_PASSWORD}@cache.internal:6379/0"


class FinalTextAfterToolLLM:
    def __init__(self, final_text: str, terminal_text: str = "") -> None:
        self.final_text = final_text
        self.terminal_text = terminal_text
        self.tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "FinalTextAfterToolLLM":
        self.tools = list(tools)
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        if messages and isinstance(messages[0], SystemMessage) and "iteration budget" in str(
            messages[0].content
        ).lower():
            return AIMessage(content=self.terminal_text)
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content=self.final_text)
        info_tool = next(tool.name for tool in self.tools if tool.name.endswith("info"))
        return AIMessage(
            content="",
            tool_calls=[{"id": "call_final_text", "name": info_tool, "args": {}}],
        )


class KnowledgeSearchLLM:
    def __init__(self) -> None:
        self.tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "KnowledgeSearchLLM":
        self.tools = list(tools)
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="Knowledge-grounded final answer")
        tool = next(
            item
            for item in self.tools
            if item.name.startswith("knowledge_") and item.name.endswith("search")
        )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "chat_knowledge_call",
                    "name": tool.name,
                    "args": {"query": "redis latency"},
                }
            ],
        )


class MCPThenRedisChatLLM:
    def __init__(self) -> None:
        self.tools: list[Any] = []
        self.bound_snapshots: list[set[str]] = []

    def bind_tools(self, tools: list[Any]) -> "MCPThenRedisChatLLM":
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
                        "id": "chat_mcp_read",
                        "name": mcp_tool.name,
                        "args": {"detail": True},
                    }
                ],
            )
        if len(tool_messages) == 1:
            info_tool = next(tool for tool in self.tools if tool.name.endswith("info"))
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "chat_redis_info",
                        "name": info_tool.name,
                        "args": {"section": "memory"},
                    }
                ],
            )
        return AIMessage(content="MCP and Redis evidence collected")


class FakeAgentMCPSession:
    def __init__(self, *, is_error: bool = False) -> None:
        self.is_error = is_error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        return mcp_types.CallToolResult(
            isError=self.is_error,
            content=[mcp_types.TextContent(type="text", text="fake external status")],
            structuredContent={"external_status": "healthy"} if not self.is_error else None,
        )


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


def test_chat_prompt_preserves_original_target_discovery_rules() -> None:
    prompt = CHAT_SYSTEM_PROMPT.lower()

    assert "list_known_redis_targets" in prompt
    assert "resolve_redis_targets" in prompt
    assert "exact live match" in prompt
    assert "ambiguous" in prompt


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
    agent = ChatAgent(
        redis_instance=make_instance(),
        llm=FakeToolCallingLLM(agent_kind="chat"),
    )

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


@pytest.mark.asyncio
async def test_chat_returns_llm_final_text_after_tool_loop(monkeypatch) -> None:
    def fake_get_client(self):
        if self._client is None:
            self._client = FakeChatRedisClient()
        return self._client

    monkeypatch.setattr(RedisCommandToolProvider, "get_client", fake_get_client)
    agent = ChatAgent(
        redis_instance=make_instance(),
        llm=FinalTextAfterToolLLM("FAKE_LLM_FINAL_ANSWER"),
    )

    response = await agent.process_query(
        "check one metric",
        session_id="session-final-text",
        user_id=None,
        context={"thread_id": "thread-final-text"},
    )

    assert response.response == "FAKE_LLM_FINAL_ANSWER"
    assert response.tool_envelopes
    assert "Redis 诊断摘要" not in response.response


@pytest.mark.asyncio
async def test_chat_executes_mcp_then_redis_and_keeps_both_top_level_envelopes(
    monkeypatch,
) -> None:
    import redis_sre_agent.tools.manager as manager_module

    sessions: list[FakeAgentMCPSession] = []

    async def fake_mcp_connect(self) -> None:
        session = FakeAgentMCPSession()
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

    def fake_get_client(self):
        if self._client is None:
            self._client = FakeChatRedisClient()
        return self._client

    monkeypatch.setattr(MCPToolProvider, "_connect", fake_mcp_connect)
    monkeypatch.setattr(RedisCommandToolProvider, "get_client", fake_get_client)
    monkeypatch.setattr(
        manager_module,
        "settings",
        Settings(
            _env_file=None,
            rag_enabled=False,
            mcp_servers={
                "agent_fake": {
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
    llm = MCPThenRedisChatLLM()
    response = await ChatAgent(
        redis_instance=make_instance(),
        llm=llm,
    ).process_query(
        "check external and redis status",
        session_id="session-chat-mcp",
        user_id=None,
        context={"thread_id": "thread-chat-mcp"},
    )

    by_name = {envelope["name"]: envelope for envelope in response.tool_envelopes}
    assert response.response == "MCP and Redis evidence collected"
    assert by_name["read_status"]["status"] == "success"
    assert by_name["info"]["status"] == "success"
    assert sessions and sessions[0].calls == [("read_status", {"detail": True})]
    assert any(any(name.startswith("mcp_") for name in snapshot) for snapshot in llm.bound_snapshots)


@pytest.mark.asyncio
async def test_chat_iteration_limit_uses_llm_terminal_synthesis(monkeypatch) -> None:
    def fake_get_client(self):
        if self._client is None:
            self._client = FakeChatRedisClient()
        return self._client

    monkeypatch.setattr(RedisCommandToolProvider, "get_client", fake_get_client)
    llm = FinalTextAfterToolLLM("unused", terminal_text="FAKE_TERMINAL_SYNTHESIS")
    agent = ChatAgent(redis_instance=make_instance(), llm=llm)

    response = await agent.process_query(
        "stop at limit",
        session_id="session-limit",
        user_id=None,
        max_iterations=1,
        context={"thread_id": "thread-limit"},
    )

    assert response.response == "FAKE_TERMINAL_SYNTHESIS"
    assert "Redis 诊断摘要" not in response.response


@pytest.mark.asyncio
async def test_chat_knowledge_search_enters_top_level_envelope_and_citation(monkeypatch) -> None:
    import redis_sre_agent.tools.manager as manager_module
    import redis_sre_agent.tools.knowledge.knowledge_base as provider_module

    monkeypatch.setattr(
        manager_module,
        "settings",
        Settings(
            _env_file=None,
            rag_enabled=True,
            embedding_api_key=SecretStr("TEST_EMBEDDING_KEY"),
        ),
    )

    async def ready(_config=None):
        return redis_core.RAGReadiness("ready", "ready", "RAG 已就绪。")

    async def knowledge_result(**_kwargs: Any) -> dict[str, Any]:
        return {
            "status": "success",
            "retrieval_kind": "knowledge_search",
            "retrieval_label": "Knowledge search",
            "results": [
                {
                    "title": "Latency runbook",
                    "source": "file://shared/latency.md",
                    "document_hash": "doc-chat",
                    "chunk_index": 0,
                    "score": 0.12,
                    "content": "Inspect SLOWLOG.",
                }
            ],
        }

    monkeypatch.setattr(redis_core, "get_rag_readiness", ready)
    monkeypatch.setattr(provider_module, "search_knowledge_base_helper", knowledge_result)
    agent = ChatAgent(llm=KnowledgeSearchLLM())

    response = await agent.process_query(
        "find latency guidance",
        session_id="session-chat-rag",
        user_id=None,
        context={"thread_id": "thread-chat-rag"},
    )

    knowledge_envelopes = [
        envelope
        for envelope in response.tool_envelopes
        if envelope["tool_key"].startswith("knowledge_")
    ]
    assert response.response == "Knowledge-grounded final answer"
    assert len(knowledge_envelopes) == 1
    assert knowledge_envelopes[0]["data"]["results"][0]["document_hash"] == "doc-chat"
    assert response.search_results == [
        {
            "title": "Latency runbook",
            "source": "file://shared/latency.md",
            "document_hash": "doc-chat",
            "chunk_index": 0,
            "score": 0.12,
            "content": "Inspect SLOWLOG.",
            "retrieval_kind": "knowledge_search",
            "retrieval_label": "Knowledge search",
        }
    ]
