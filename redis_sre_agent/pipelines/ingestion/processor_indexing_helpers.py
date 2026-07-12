"""把一个已切分文档写入唯一的 knowledge deduplicator。"""

from __future__ import annotations

from typing import Any, Dict, List

from redis_sre_agent.pipelines.scraper.base import ScrapedDocument

from .deduplication import DocumentDeduplicator


def get_source_tracking_fields(document: ScrapedDocument) -> tuple[str, str]:
    return (
        str(document.metadata.get("source_document_path") or "").strip(),
        str(document.metadata.get("source_document_scope") or "").strip(),
    )


async def index_processed_document(
    *,
    document: ScrapedDocument,
    chunks: List[Dict[str, Any]],
    vectorizer: Any,
    deduplicator: DocumentDeduplicator,
) -> Dict[str, Any]:
    source_path, source_scope = get_source_tracking_fields(document)
    if source_path:
        replacement = await deduplicator.replace_source_document_chunks(chunks, vectorizer)
        action = str(replacement.get("action") or "unchanged")
        indexed_count = int(replacement.get("indexed_count") or 0)
    else:
        indexed_count = await deduplicator.replace_document_chunks(chunks, vectorizer)
        action = "add" if indexed_count else "unchanged"
    return {
        "chunks_created": len(chunks),
        "chunks_indexed": indexed_count,
        "source_document_change": {
            "path": source_path,
            "action": action,
            "title": document.title,
            "doc_type": document.doc_type.value,
        },
        "source_document_path": source_path,
        "source_document_scope": source_scope,
    }
