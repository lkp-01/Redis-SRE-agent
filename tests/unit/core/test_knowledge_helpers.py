"""最小 RedisVL 向量检索 helper 测试。"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from pydantic import SecretStr
from redisvl.query import VectorQuery, VectorRangeQuery

from redis_sre_agent.core import knowledge_helpers
from redis_sre_agent.core.config import Settings
from redis_sre_agent.core.redis import RAGReadiness


class FakeVectorizer:
    def __init__(self) -> None:
        self.values: list[list[str]] = []

    async def aembed(self, value: str, **_kwargs: Any) -> list[float]:
        return (await self.aembed_many([value]))[0]

    async def aembed_many(self, values: list[str], **_kwargs: Any) -> list[list[float]]:
        self.values.append(list(values))
        return [[1.0, 0.0, 0.0] for _ in values]


class FakeClient:
    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


class FakeIndex:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.queries: list[Any] = []
        self._redis_client = FakeClient()
        self.client = self._redis_client

    async def query(self, query: Any) -> list[dict[str, Any]]:
        self.queries.append(query)
        return self.results


def _config() -> Settings:
    return Settings(
        _env_file=None,
        rag_enabled=True,
        embedding_provider="openai",
        embedding_api_key=SecretStr("TEST_EMBEDDING_KEY"),
        vector_dim=3,
    )


def _result(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": "sre_knowledge:doc:chunk:2",
        "document_hash": "doc",
        "chunk_index": "2",
        "title": "Latency runbook",
        "content": "Inspect SLOWLOG.",
        "source": "file://shared/latency.md",
        "category": "shared",
        "doc_type": "runbook",
        "name": "latency",
        "summary": "Latency diagnosis",
        "priority": "high",
        "pinned": "false",
        "version": "latest",
        "vector_distance": "0.125",
    }
    value.update(overrides)
    return value


async def _patch_ready(monkeypatch, index: FakeIndex, vectorizer: FakeVectorizer) -> None:
    async def ready(_config=None):
        return RAGReadiness("ready", "ready", "RAG 已就绪。")

    async def get_index(_config=None):
        return index

    monkeypatch.setattr(knowledge_helpers, "get_rag_readiness", ready)
    monkeypatch.setattr(knowledge_helpers, "get_knowledge_index", get_index)
    monkeypatch.setattr(knowledge_helpers, "get_vectorizer", lambda _config=None: vectorizer)


@pytest.mark.asyncio
async def test_range_query_embeds_filters_pages_and_normalizes_score(monkeypatch) -> None:
    index = FakeIndex([_result()])
    vectorizer = FakeVectorizer()
    await _patch_ready(monkeypatch, index, vectorizer)

    result = await knowledge_helpers.search_knowledge_base_helper(
        "redis latency",
        category="shared",
        limit=2,
        offset=3,
        distance_threshold=0.8,
        version="latest",
        config=_config(),
    )

    assert vectorizer.values == [["redis latency"]]
    assert len(index.queries) == 1
    query = index.queries[0]
    assert isinstance(query, VectorRangeQuery)
    assert query._distance_threshold == 0.8
    assert query._num_results == 5
    assert query._offset == 3
    assert query._num == 2
    rendered = str(query)
    assert "@version:{latest}" in rendered
    assert "@category:{shared}" in rendered
    assert result["status"] == "success"
    assert result["results_count"] == 1
    assert result["retrieval_kind"] == "knowledge_search"
    assert result["retrieval_label"] == "Knowledge search"
    assert result["results"][0]["score"] == 0.125
    assert result["results"][0]["chunk_index"] == 2
    assert index.client.closed == 1


@pytest.mark.asyncio
async def test_none_threshold_uses_vector_query_and_empty_match_is_success(monkeypatch) -> None:
    index = FakeIndex([])
    vectorizer = FakeVectorizer()
    await _patch_ready(monkeypatch, index, vectorizer)

    result = await knowledge_helpers.search_knowledge_base_helper(
        "memory",
        limit="0",
        offset="-2",
        distance_threshold=None,
        version=None,
        config=_config(),
    )

    query = index.queries[0]
    assert isinstance(query, VectorQuery)
    assert query._num_results == 1
    assert query._offset == 0
    assert query._num == 1
    assert "@version" not in str(query)
    assert result["status"] == "success"
    assert result["results"] == []
    assert result["results_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "reason_code"),
    [
        ("disabled", "disabled"),
        ("not_ready", "index_missing"),
    ],
)
async def test_unavailable_rag_returns_explicit_error_without_querying(
    monkeypatch,
    state: str,
    reason_code: str,
) -> None:
    async def readiness(_config=None):
        return RAGReadiness(state, reason_code, "安全原因。")

    monkeypatch.setattr(knowledge_helpers, "get_rag_readiness", readiness)
    monkeypatch.setattr(
        knowledge_helpers,
        "get_vectorizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unavailable RAG must not construct vectorizer")
        ),
    )

    result = await knowledge_helpers.search_knowledge_base_helper(
        "memory",
        config=Settings(_env_file=None, rag_enabled=(state != "disabled")),
    )

    assert result["status"] == "unavailable"
    assert result["reason_code"] == reason_code
    assert result["results"] == []


def test_minimum_helper_does_not_restore_hybrid_or_scan_fallback() -> None:
    source = inspect.getsource(knowledge_helpers)
    assert "HybridQuery" not in source
    assert "RRF" not in source
    assert "scan_iter" not in source
    assert "skills" not in source.lower()
    assert "support_ticket" not in source
