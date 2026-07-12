"""本地 Markdown 身份与 original 风格切分规则测试。"""

from __future__ import annotations

from pathlib import Path

from redis_sre_agent.pipelines.ingestion.document_processor import DocumentProcessor
from redis_sre_agent.pipelines.ingestion.processor_source_helpers import (
    create_scraped_document_from_markdown,
)
from redis_sre_agent.pipelines.scraper.base import (
    DocumentCategory,
    DocumentType,
    ScrapedDocument,
)


def _document(content: str) -> ScrapedDocument:
    return ScrapedDocument(
        title="Latency runbook",
        content=content,
        source_url="test://latency",
        category=DocumentCategory.SHARED,
        doc_type=DocumentType.RUNBOOK,
    )


def test_scraped_document_hash_is_deterministic() -> None:
    first = _document("Inspect SLOWLOG.")
    second = _document("Inspect SLOWLOG.")

    assert first.document_hash == second.document_hash
    assert first.content_hash == second.content_hash
    assert len(first.document_hash) == 16


def test_markdown_source_identity_is_stable_across_checkout_roots(tmp_path: Path) -> None:
    contents = """---
title: Latency runbook
category: shared
doc_type: runbook
severity: high
priority: high
pinned: true
version: latest
---
# Latency runbook

Inspect SLOWLOG before changing configuration.
"""
    documents = []
    for root_name in ("checkout-a", "checkout-b"):
        root = tmp_path / root_name / "source_documents"
        path = root / "shared" / "latency.md"
        path.parent.mkdir(parents=True)
        path.write_text(contents, encoding="utf-8")
        documents.append(create_scraped_document_from_markdown(path, root))

    first, second = documents
    assert first.source_url == "file://shared/latency.md"
    assert second.source_url == first.source_url
    assert first.metadata["source_document_path"] == "shared/latency.md"
    assert first.metadata["source_document_scope"] == ""
    assert first.document_hash == second.document_hash
    assert first.category is DocumentCategory.SHARED
    assert first.doc_type is DocumentType.RUNBOOK
    assert first.metadata["priority"] == "high"
    assert first.metadata["pinned"] is True


def test_short_document_is_kept_as_one_chunk_even_below_minimum() -> None:
    chunks = DocumentProcessor().chunk_document(_document("Short note."))

    assert len(chunks) == 1
    assert chunks[0]["content"] == "Short note."
    assert chunks[0]["chunk_index"] == 0


def test_front_matter_is_removed_before_chunking() -> None:
    content = "---\ntitle: Hidden metadata\n---\n# Visible\n\nCheck INFO memory."

    chunks = DocumentProcessor().chunk_document(_document(content))

    assert len(chunks) == 1
    assert chunks[0]["content"].startswith("# Visible")
    assert "Hidden metadata" not in chunks[0]["content"]


def test_long_document_prefers_sentence_boundary_and_overlaps() -> None:
    first_sentence = "A" * 850 + "."
    tail = "B" * 900
    processor = DocumentProcessor(
        {"chunk_size": 1000, "chunk_overlap": 200, "min_chunk_size": 100}
    )

    chunks = processor.chunk_document(_document(first_sentence + " " + tail))

    assert len(chunks) >= 2
    assert chunks[0]["content"].endswith(".")
    assert chunks[1]["content"].startswith("A" * 199 + ".")
    assert all(len(chunk["content"]) <= 1000 for chunk in chunks)


def test_trailing_fragment_below_minimum_is_not_indexed() -> None:
    processor = DocumentProcessor(
        {"chunk_size": 300, "chunk_overlap": 0, "min_chunk_size": 100}
    )
    content = "A" * 300 + " " + "B" * 50

    chunks = processor.chunk_document(_document(content))

    assert len(chunks) == 1
    assert len(chunks[0]["content"]) == 300
