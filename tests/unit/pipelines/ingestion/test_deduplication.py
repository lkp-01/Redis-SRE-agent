"""knowledge key 与 source tracking 的确定性测试。"""

from __future__ import annotations

from redis_sre_agent.pipelines.ingestion.deduplication import DocumentDeduplicator


def test_deterministic_knowledge_keys_match_original_contract() -> None:
    deduplicator = DocumentDeduplicator(index=object())

    assert (
        deduplicator.generate_deterministic_chunk_key("abc123", 2)
        == "sre_knowledge:abc123:chunk:2"
    )
    assert (
        deduplicator.generate_document_tracking_key("abc123")
        == "sre_knowledge_meta:abc123"
    )
    first = deduplicator.generate_source_tracking_key("shared/latency.md")
    second = deduplicator.generate_source_tracking_key("shared/latency.md")
    assert first == second
    assert first.startswith("sre_knowledge_meta:source:")


def test_prepare_chunks_uses_deterministic_ids() -> None:
    deduplicator = DocumentDeduplicator(index=object())
    chunks = [
        {"document_hash": "abc123", "chunk_index": 0, "content": "one"},
        {"document_hash": "abc123", "chunk_index": 1, "content": "two"},
    ]

    prepared = deduplicator.prepare_chunks_for_replacement(chunks)

    assert [item["id"] for item in prepared] == [
        "sre_knowledge:abc123:chunk:0",
        "sre_knowledge:abc123:chunk:1",
    ]
    assert [item["chunk_key"] for item in prepared] == [
        "sre_knowledge:abc123:chunk:0",
        "sre_knowledge:abc123:chunk:1",
    ]
