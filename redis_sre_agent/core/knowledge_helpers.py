"""knowledge index 的最小纯向量检索 helper。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from redisvl.query import VectorQuery, VectorRangeQuery
from redisvl.query.filter import Tag

from redis_sre_agent.core.config import Settings, settings
from redis_sre_agent.core.redis import (
    _close_index_client,
    get_knowledge_index,
    get_rag_readiness,
    get_vectorizer,
)

_SEARCH_RETURN_FIELDS = [
    "id",
    "document_hash",
    "chunk_index",
    "title",
    "content",
    "source",
    "category",
    "doc_type",
    "name",
    "summary",
    "priority",
    "pinned",
    "version",
]


def _coerce_non_negative_int(value: Any, *, default: int) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def _coerce_positive_int(value: Any, *, default: int) -> int:
    return max(_coerce_non_negative_int(value, default=default), 1)


def _decode(value: Any) -> Any:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _normalized_chunk_index(value: Any) -> Any:
    if value in (None, ""):
        return None
    try:
        return int(_decode(value))
    except (TypeError, ValueError):
        return _decode(value)


def _normalized_score(document: Dict[str, Any]) -> float:
    value = document.get("score")
    if value is None:
        value = document.get("vector_distance")
    if value is None:
        value = document.get("distance")
    try:
        return float(_decode(value)) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _unavailable_result(
    *,
    query: str,
    reason_code: str,
    message: str,
) -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "reason_code": reason_code,
        "message": message,
        "query": query,
        "retrieval_kind": "knowledge_search",
        "retrieval_label": "Knowledge search",
        "results_count": 0,
        "results": [],
    }


async def search_knowledge_base_helper(
    query: str,
    category: Optional[str] = None,
    doc_type: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    distance_threshold: Optional[float] = 0.8,
    version: Optional[str] = "latest",
    config: Optional[Settings] = None,
) -> Dict[str, Any]:
    """用 query embedding 在唯一 knowledge index 中执行向量检索。"""

    cfg = config or settings
    readiness = await get_rag_readiness(cfg)
    if not readiness.ready:
        return _unavailable_result(
            query=query,
            reason_code=readiness.reason_code,
            message=readiness.message,
        )

    normalized_limit = min(_coerce_positive_int(limit, default=10), 50)
    normalized_offset = _coerce_non_negative_int(offset, default=0)
    fetch_limit = normalized_limit + normalized_offset

    try:
        vectorizer = get_vectorizer(cfg)
        vectors = await vectorizer.aembed_many([query])
        query_vector = vectors[0] if vectors else []
        if len(query_vector) != cfg.vector_dim:
            return _unavailable_result(
                query=query,
                reason_code="embedding_config_invalid",
                message="query embedding 维度与 knowledge index 不一致。",
            )
    except Exception:
        return _unavailable_result(
            query=query,
            reason_code="embedding_unavailable",
            message="embedding provider 当前不可用。",
        )

    filter_expression = None
    if version is not None:
        filter_expression = Tag("version") == version
    if category is not None:
        category_filter = Tag("category") == category
        filter_expression = (
            category_filter
            if filter_expression is None
            else filter_expression & category_filter
        )
    if doc_type is not None:
        type_filter = Tag("doc_type") == doc_type
        filter_expression = (
            type_filter if filter_expression is None else filter_expression & type_filter
        )

    query_kwargs = {
        "vector": query_vector,
        "vector_field_name": "vector",
        "return_fields": list(_SEARCH_RETURN_FIELDS),
        "filter_expression": filter_expression,
        "num_results": fetch_limit,
    }
    if distance_threshold is None:
        vector_query: Any = VectorQuery(**query_kwargs)
    else:
        vector_query = VectorRangeQuery(
            **query_kwargs,
            distance_threshold=float(distance_threshold),
        )
    vector_query.paging(normalized_offset, normalized_limit)

    index = None
    try:
        index = await get_knowledge_index(cfg)
        raw_results = await index.query(vector_query)
    except Exception:
        return _unavailable_result(
            query=query,
            reason_code="redis_search_unavailable",
            message="Redis Search/Vector 查询失败。",
        )
    finally:
        if index is not None:
            await _close_index_client(index)

    results: List[Dict[str, Any]] = []
    for raw_document in raw_results or []:
        document = {
            str(_decode(key)): _decode(value)
            for key, value in dict(raw_document).items()
        }
        results.append(
            {
                "id": document.get("id", ""),
                "document_hash": document.get("document_hash", ""),
                "chunk_index": _normalized_chunk_index(document.get("chunk_index")),
                "title": document.get("title", ""),
                "content": document.get("content", ""),
                "source": document.get("source", ""),
                "category": document.get("category", ""),
                "doc_type": document.get("doc_type", "knowledge"),
                "name": document.get("name", ""),
                "summary": document.get("summary", ""),
                "priority": document.get("priority", "normal"),
                "pinned": document.get("pinned", "false"),
                "version": document.get("version", "latest"),
                "score": _normalized_score(document),
            }
        )
    return {
        "status": "success",
        "reason_code": "ready",
        "query": query,
        "category": category,
        "doc_type": doc_type,
        "version": version,
        "offset": normalized_offset,
        "limit": normalized_limit,
        "distance_threshold": distance_threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieval_kind": "knowledge_search",
        "retrieval_label": "Knowledge search",
        "results_count": len(results),
        "results": results,
    }
