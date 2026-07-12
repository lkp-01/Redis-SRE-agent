"""显式 opt-in 的 Redis Stack 向量索引集成测试。"""

from __future__ import annotations

import os
from array import array
from uuid import uuid4

import pytest


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_ft_create_hash_write_and_vector_query() -> None:
    if os.getenv("RUN_RAG_REDIS_INTEGRATION") != "1":
        pytest.skip("set RUN_RAG_REDIS_INTEGRATION=1 to run Redis Stack integration")

    redis_url = os.getenv("RAG_REDIS_TEST_URL")
    if not redis_url:
        pytest.skip("RAG_REDIS_TEST_URL must point to an isolated Redis Stack instance")

    from redis.asyncio import Redis
    from redisvl.index import AsyncSearchIndex
    from redisvl.query import VectorQuery
    from redisvl.schema import IndexSchema

    from redis_sre_agent.core.redis import _build_document_schema

    suffix = uuid4().hex
    index_name = f"sre_knowledge_test_{suffix}"
    schema_dict = _build_document_schema(
        index_name,
        include_pinned=True,
        vector_dim=3,
    )
    client = Redis.from_url(redis_url, decode_responses=False)
    index = AsyncSearchIndex(
        schema=IndexSchema.from_dict(schema_dict),
        redis_client=client,
    )
    key = f"{index_name}:doc:chunk:0"

    try:
        await index.create()
        await client.hset(
            key,
            mapping={
                "id": key,
                "document_hash": "doc",
                "content_hash": "content",
                "title": "Latency runbook",
                "content": "Inspect SLOWLOG before changing configuration.",
                "source": "test://runbook",
                "category": "shared",
                "doc_type": "runbook",
                "name": "latency-runbook",
                "summary": "Latency diagnosis",
                "priority": "normal",
                "pinned": "false",
                "severity": "medium",
                "product_labels": "redis",
                "product_label_tags": "redis",
                "version": "latest",
                "chunk_index": 0,
                "created_at": 1,
                "vector": array("f", [1.0, 0.0, 0.0]).tobytes(),
            },
        )
        query = VectorQuery(
            vector=[1.0, 0.0, 0.0],
            vector_field_name="vector",
            return_fields=["title", "source", "document_hash", "chunk_index"],
            num_results=1,
        )

        results = await index.query(query)

        assert len(results) == 1
        assert results[0]["source"] == "test://runbook"
        assert results[0]["document_hash"] == "doc"
        assert float(results[0].get("vector_distance", results[0].get("score", 0.0))) >= 0.0
    finally:
        try:
            await client.execute_command("FT.DROPINDEX", index_name, "DD")
        finally:
            await client.aclose()
