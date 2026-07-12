"""Fake runtime 下的 RAG evidence 响应与 CLI 闭环测试。"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import SecretStr

from redis_sre_agent.agent.chat_agent import ChatAgent
from redis_sre_agent.agent.models import AgentResponse
from redis_sre_agent.cli.main import main
from redis_sre_agent.core import knowledge_helpers, redis as redis_core
from redis_sre_agent.core.config import Settings
from redis_sre_agent.pipelines.ingestion.processor import IngestionPipeline
from tests.support.fake_redis import FakeRedis


def _envelope() -> dict[str, Any]:
    return {
        "tool_key": "knowledge_abcdef_search",
        "name": "search",
        "description": "搜索 knowledge base。",
        "args": {"query": "latency"},
        "status": "success",
        "data": {
            "status": "success",
            "retrieval_kind": "knowledge_search",
            "retrieval_label": "Knowledge search",
            "results": [
                {
                    "title": "Latency runbook",
                    "source": "file://shared/latency.md",
                    "document_hash": "doc-cli",
                    "chunk_index": 0,
                    "score": 0.1,
                    "content": "Inspect SLOWLOG.",
                }
            ],
        },
        "summary": None,
    }


def test_agent_response_ignores_forged_search_results_and_derives_from_top_level_envelopes() -> None:
    response = AgentResponse(
        response="grounded",
        search_results=[{"source": "forged"}],
        tool_envelopes=[_envelope()],
    )

    assert response.search_results == [
        {
            "title": "Latency runbook",
            "source": "file://shared/latency.md",
            "document_hash": "doc-cli",
            "chunk_index": 0,
            "score": 0.1,
            "content": "Inspect SLOWLOG.",
            "retrieval_kind": "knowledge_search",
            "retrieval_label": "Knowledge search",
        }
    ]
    assert AgentResponse(response="none", search_results=[{"source": "forged"}]).search_results == []


def test_query_cli_json_contains_response_citations_envelopes_and_thread(monkeypatch) -> None:
    query_module = importlib.import_module("redis_sre_agent.cli.query")
    redis = FakeRedis()

    class StubAgent:
        async def process_query(self, *_args: Any, **_kwargs: Any) -> AgentResponse:
            return AgentResponse(response="grounded", tool_envelopes=[_envelope()])

    monkeypatch.setattr(query_module, "get_redis_client", lambda: redis)
    monkeypatch.setattr(query_module, "get_chat_agent", lambda **_kwargs: StubAgent())

    result = CliRunner().invoke(main, ["query", "latency guidance", "--agent", "chat"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["response"] == "grounded"
    assert payload["search_results"][0]["document_hash"] == "doc-cli"
    assert payload["tool_envelopes"][0]["tool_key"] == "knowledge_abcdef_search"
    assert payload["thread_id"]


class ClosedLoopIndex:
    def __init__(self, client: FakeRedis) -> None:
        self.client = client
        self._redis_client = client
        self.queries: list[Any] = []

    async def query(self, query: Any) -> list[dict[str, Any]]:
        self.queries.append(query)
        documents = []
        for key, mapping in self.client.hashes.items():
            if str(key).startswith("sre_knowledge:") and ":chunk:" in str(key):
                document = dict(mapping)
                document["vector_distance"] = "0.0"
                documents.append(document)
        return documents[:1]


class ClosedLoopVectorizer:
    async def aembed(self, _value: str, **_kwargs: Any) -> list[float]:
        return [1.0, 0.0, 0.0]

    async def aembed_many(self, values: list[str], **_kwargs: Any) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in values]


class ClosedLoopKnowledgeLLM:
    def __init__(self) -> None:
        self.tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "ClosedLoopKnowledgeLLM":
        self.tools = list(tools)
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="Use the cited latency runbook.")
        tool = next(
            item
            for item in self.tools
            if item.name.startswith("knowledge_") and item.name.endswith("search")
        )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "closed-loop-search",
                    "name": tool.name,
                    "args": {"query": "latency"},
                }
            ],
        )


@pytest.mark.asyncio
async def test_direct_ingest_to_toolmanager_chat_citation_closed_loop(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import redis_sre_agent.tools.manager as manager_module

    source_dir = tmp_path / "source_documents"
    markdown = source_dir / "shared" / "latency.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text(
        "---\n"
        "title: Latency runbook\n"
        "category: shared\n"
        "doc_type: runbook\n"
        "version: latest\n"
        "---\n"
        "# Latency runbook\n\nInspect SLOWLOG before changing configuration.\n",
        encoding="utf-8",
    )
    config = Settings(
        _env_file=None,
        rag_enabled=True,
        embedding_provider="custom",
        vectorizer_factory="tests.fake.closed_loop_vectorizer",
        vector_dim=3,
    )
    redis = FakeRedis()
    index = ClosedLoopIndex(redis)
    vectorizer = ClosedLoopVectorizer()
    pipeline = IngestionPipeline(
        settings_config=config,
        index=index,
        vectorizer=vectorizer,
    )

    ingested = await pipeline.ingest_source_documents(source_dir)
    assert ingested[0]["status"] == "success"
    assert ingested[0]["chunks_indexed"] == 1

    async def ready(_config=None):
        return redis_core.RAGReadiness("ready", "ready", "RAG 已就绪。")

    async def get_index(_config=None):
        return index

    monkeypatch.setattr(manager_module, "settings", config)
    monkeypatch.setattr(knowledge_helpers, "settings", config)
    monkeypatch.setattr(redis_core, "get_rag_readiness", ready)
    monkeypatch.setattr(knowledge_helpers, "get_rag_readiness", ready)
    monkeypatch.setattr(knowledge_helpers, "get_knowledge_index", get_index)
    monkeypatch.setattr(knowledge_helpers, "get_vectorizer", lambda _config=None: vectorizer)

    response = await ChatAgent(llm=ClosedLoopKnowledgeLLM()).process_query(
        "find latency guidance",
        session_id="closed-loop",
        user_id=None,
        context={"thread_id": "closed-loop-thread"},
    )

    assert index.queries
    assert response.response == "Use the cited latency runbook."
    assert len(response.tool_envelopes) == 1
    assert response.search_results[0]["title"] == "Latency runbook"
    assert response.search_results[0]["source"] == "file://shared/latency.md"
    assert response.search_results[0]["retrieval_kind"] == "knowledge_search"
