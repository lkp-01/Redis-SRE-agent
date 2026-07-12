"""本地 Markdown 的顺序摄取实现，不包含 scraper、skills 或多索引逻辑。"""

from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from redis_sre_agent.core.config import Settings, settings
from redis_sre_agent.core.redis import RAGNotReadyError

from .deduplication import DocumentDeduplicator
from .document_processor import DocumentProcessor
from .pipeline_workflow_mixin import PipelineWorkflowMixin
from .processor_indexing_helpers import index_processed_document
from .processor_source_helpers import (
    create_scraped_document_from_markdown,
    find_markdown_files,
)

logger = logging.getLogger(__name__)


async def get_knowledge_index(config: Optional[Settings] = None):
    from redis_sre_agent.core.redis import get_knowledge_index as _get_knowledge_index

    return await _get_knowledge_index(config=config)


async def ensure_knowledge_index(
    config: Optional[Settings] = None,
    *,
    create_if_missing: bool,
):
    from redis_sre_agent.core.redis import ensure_knowledge_index as _ensure

    return await _ensure(config=config, create_if_missing=create_if_missing)


def get_vectorizer(config: Optional[Settings] = None):
    from redis_sre_agent.core.redis import get_vectorizer as _get_vectorizer

    return _get_vectorizer(config=config)


class IngestionPipeline(PipelineWorkflowMixin):
    """先显式确保索引，再逐个处理本地 Markdown 文档。"""

    def __init__(
        self,
        storage: Any = None,
        config: Optional[Dict[str, Any]] = None,
        knowledge_settings: Any = None,
        *,
        settings_config: Optional[Settings] = None,
        index: Any = None,
        vectorizer: Any = None,
    ) -> None:
        self.storage = storage
        self.config = config or {}
        self.knowledge_settings = knowledge_settings
        self.settings = settings_config or settings
        self.processor = DocumentProcessor(self.config, knowledge_settings)
        self._index = index
        self._vectorizer = vectorizer

    async def _resolve_runtime(self) -> tuple[Any, Any]:
        if not self.settings.rag_enabled:
            raise RAGNotReadyError("disabled", "RAG 未启用。")
        index = self._index
        if index is None:
            index = await ensure_knowledge_index(
                self.settings,
                create_if_missing=True,
            )
        vectorizer = self._vectorizer
        if vectorizer is None:
            vectorizer = get_vectorizer(self.settings)
        return index, vectorizer

    async def ingest_source_documents(self, source_dir: Path | str) -> List[Dict[str, Any]]:
        source_path = Path(source_dir)
        if not source_path.exists() or not source_path.is_dir():
            raise ValueError("source directory 不存在或不是目录。")
        index, vectorizer = await self._resolve_runtime()
        deduplicator = DocumentDeduplicator(
            index,
            key_prefix="sre_knowledge",
            vector_dim=self.settings.vector_dim,
        )
        results: List[Dict[str, Any]] = []
        for markdown_file in find_markdown_files(source_path):
            relative_name = markdown_file.relative_to(source_path).as_posix()
            try:
                document = create_scraped_document_from_markdown(
                    markdown_file,
                    source_path,
                )
                chunks = self.processor.chunk_document(document)
                indexed = await index_processed_document(
                    document=document,
                    chunks=chunks,
                    vectorizer=vectorizer,
                    deduplicator=deduplicator,
                )
                change = indexed["source_document_change"]
                results.append(
                    {
                        "file": relative_name,
                        "title": document.title,
                        "category": document.category.value,
                        "severity": document.severity.value,
                        "status": "success",
                        "action": change["action"],
                        "chunks_created": indexed["chunks_created"],
                        "chunks_indexed": indexed["chunks_indexed"],
                        "document_hash": document.document_hash,
                    }
                )
            except Exception as exc:
                # 外部 provider/Redis 异常文本可能带连接信息，用户输出只保留安全错误码。
                logger.warning("本地 Markdown 摄取失败：%s", type(exc).__name__)
                results.append(
                    {
                        "file": relative_name,
                        "status": "error",
                        "error": "文档摄取失败。",
                        "error_type": type(exc).__name__,
                    }
                )
        return results

    async def ingest_batch(self, batch_date: str) -> Dict[str, Any]:
        """顺序摄取一个本地 artifact batch，不做并行或 stale-scope 清理。"""

        if self.storage is None:
            raise ValueError("prepared batch 摄取需要 ArtifactStorage。")
        manifest = self.storage.get_batch_manifest(batch_date)
        if not manifest:
            raise ValueError(f"No manifest found for batch {batch_date}")
        batch_path = self.storage.base_path / batch_date
        if not batch_path.exists():
            raise ValueError(f"Batch directory not found: {batch_date}")
        self.storage.set_batch_date(batch_date)

        stats: Dict[str, Any] = {
            "batch_date": batch_date,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "documents_processed": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "categories_processed": {},
            "errors": [],
            "success": False,
        }
        index, vectorizer = await self._resolve_runtime()
        deduplicator = DocumentDeduplicator(
            index,
            key_prefix="sre_knowledge",
            vector_dim=self.settings.vector_dim,
        )

        for category in ("oss", "enterprise", "shared"):
            category_path = batch_path / category
            category_stats = {
                "documents_processed": 0,
                "chunks_created": 0,
                "chunks_indexed": 0,
                "errors": [],
            }
            if category_path.exists():
                for artifact_path in sorted(category_path.glob("*.json")):
                    try:
                        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                        if not isinstance(payload, dict):
                            raise ValueError("artifact 必须是 JSON object。")
                        from redis_sre_agent.pipelines.scraper.base import ScrapedDocument

                        document = ScrapedDocument.from_dict(payload)
                        chunks = self.processor.chunk_document(document)
                        indexed = await index_processed_document(
                            document=document,
                            chunks=chunks,
                            vectorizer=vectorizer,
                            deduplicator=deduplicator,
                        )
                        category_stats["documents_processed"] += 1
                        category_stats["chunks_created"] += indexed["chunks_created"]
                        category_stats["chunks_indexed"] += indexed["chunks_indexed"]
                    except Exception as exc:
                        category_stats["errors"].append(
                            {
                                "file": artifact_path.name,
                                "error": "artifact 摄取失败。",
                                "error_type": type(exc).__name__,
                            }
                        )
            stats["categories_processed"][category] = category_stats
            stats["documents_processed"] += category_stats["documents_processed"]
            stats["chunks_created"] += category_stats["chunks_created"]
            stats["chunks_indexed"] += category_stats["chunks_indexed"]
            stats["errors"].extend(category_stats["errors"])

        stats["completed_at"] = datetime.now(timezone.utc).isoformat()
        stats["success"] = not stats["errors"]
        self.storage.save_ingestion_manifest(batch_date, stats)
        return stats
