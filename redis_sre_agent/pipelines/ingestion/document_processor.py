"""将一个本地文档切成可写入向量索引的稳定分块。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from redis_sre_agent.pipelines.scraper.base import ScrapedDocument

from .processor_source_helpers import parse_bool, strip_yaml_front_matter


class DocumentProcessor:
    """保留 original 的字符切分、句号/空格边界和 overlap 规则。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, knowledge_settings=None):
        source = config or {}
        if knowledge_settings is not None:
            source = {
                "chunk_size": knowledge_settings.chunk_size,
                "chunk_overlap": knowledge_settings.chunk_overlap,
                "max_chunks_per_doc": knowledge_settings.max_documents_per_batch,
            }
        self.config: Dict[str, Any] = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "min_chunk_size": 100,
            "max_chunks_per_doc": 10,
            "strip_front_matter": True,
            **source,
        }
        chunk_size = int(self.config["chunk_size"])
        overlap = int(self.config["chunk_overlap"])
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size 必须为正数，且 chunk_overlap 必须小于 chunk_size。")

    def chunk_document(self, document: ScrapedDocument) -> List[Dict[str, Any]]:
        content = document.content or ""
        if self.config.get("strip_front_matter", True):
            content, _ = strip_yaml_front_matter(content)
        if not content.strip():
            return []
        if len(content) <= int(self.config["chunk_size"]):
            return [self._create_chunk(document, content.strip(), 0)]

        chunks: List[Dict[str, Any]] = []
        chunk_size = int(self.config["chunk_size"])
        overlap = int(self.config["chunk_overlap"])
        minimum = int(self.config["min_chunk_size"])
        maximum = int(self.config["max_chunks_per_doc"])
        start = 0
        chunk_index = 0
        while start < len(content) and chunk_index < maximum:
            end = min(start + chunk_size, len(content))
            if end < len(content):
                sentence_break = content.rfind(".", start, end)
                if sentence_break > start + chunk_size // 2:
                    end = sentence_break + 1
                else:
                    word_break = content.rfind(" ", start, end)
                    if word_break > start + chunk_size // 2:
                        end = word_break
            chunk_content = content[start:end].strip()
            if len(chunk_content) >= minimum:
                chunks.append(self._create_chunk(document, chunk_content, chunk_index))
                chunk_index += 1
            if end >= len(content):
                break
            next_start = end - overlap
            if next_start <= start:
                break
            start = next_start
        return chunks

    def _create_chunk(
        self,
        document: ScrapedDocument,
        content: str,
        chunk_index: int,
    ) -> Dict[str, Any]:
        title = (
            document.title
            if chunk_index == 0
            else f"{document.title} (Part {chunk_index + 1})"
        )
        metadata = dict(document.metadata)
        return {
            "id": f"{document.document_hash}_{chunk_index}",
            "document_hash": document.document_hash,
            "title": title,
            "content": content,
            "source": document.source_url,
            "category": document.category.value,
            "doc_type": document.doc_type.value,
            "name": str(metadata.get("name") or document.title),
            "summary": str(metadata.get("summary") or ""),
            "priority": str(metadata.get("priority") or "normal").lower(),
            "pinned": "true" if parse_bool(metadata.get("pinned")) else "false",
            "severity": document.severity.value,
            "product_labels": metadata.get("product_labels", ""),
            "product_label_tags": metadata.get("product_label_tags", ""),
            "version": str(metadata.get("version") or "latest"),
            "chunk_index": chunk_index,
            "source_document_path": str(metadata.get("source_document_path") or ""),
            "source_document_scope": str(metadata.get("source_document_scope") or ""),
            "metadata": {
                **metadata,
                "original_title": document.title,
                "chunk_size": len(content),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            },
        }
