"""单一 knowledge 索引的确定性、原子文档替换。"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from array import array
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from redis_sre_agent.core.config import settings


class DocumentDeduplicator:
    """先准备全部向量，再用一个 MULTI/EXEC 同时替换 chunk 和 tracking。"""

    def __init__(
        self,
        index: Any,
        key_prefix: str = "sre_knowledge",
        *,
        vector_dim: Optional[int] = None,
    ) -> None:
        self.index = index
        self.key_prefix = key_prefix
        self.meta_prefix = f"{key_prefix}_meta"
        self.source_meta_prefix = f"{self.meta_prefix}:source"
        self.vector_dim = int(vector_dim or settings.vector_dim)

    @property
    def client(self) -> Any:
        client = getattr(self.index, "client", None) or getattr(
            self.index, "_redis_client", None
        )
        if client is None:
            raise RuntimeError("knowledge index 没有可用 Redis client。")
        return client

    def generate_deterministic_chunk_key(self, document_hash: str, chunk_index: int) -> str:
        return f"{self.key_prefix}:{document_hash}:chunk:{chunk_index}"

    def generate_document_tracking_key(self, document_hash: str) -> str:
        return f"{self.meta_prefix}:{document_hash}"

    def generate_source_tracking_key(self, source_document_path: str) -> str:
        path_hash = hashlib.sha256(source_document_path.encode("utf-8")).hexdigest()[:16]
        return f"{self.source_meta_prefix}:{path_hash}"

    @staticmethod
    def _decode_mapping(mapping: Dict[Any, Any]) -> Dict[str, Any]:
        return {
            key.decode("utf-8") if isinstance(key, bytes) else str(key): (
                value.decode("utf-8") if isinstance(value, bytes) else value
            )
            for key, value in mapping.items()
        }

    async def find_existing_chunks(self, document_hash: str) -> List[str]:
        keys: List[str] = []
        pattern = f"{self.key_prefix}:{document_hash}:chunk:*"
        async for key in self.client.scan_iter(match=pattern):
            keys.append(key.decode("utf-8") if isinstance(key, bytes) else str(key))
        return sorted(set(keys))

    async def get_source_document_tracking(
        self,
        source_document_path: str,
    ) -> Optional[Dict[str, Any]]:
        mapping = await self.client.hgetall(
            self.generate_source_tracking_key(source_document_path)
        )
        return self._decode_mapping(mapping) if mapping else None

    def prepare_chunks_for_replacement(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [
            {
                **chunk,
                "id": self.generate_deterministic_chunk_key(
                    str(chunk["document_hash"]), int(chunk["chunk_index"])
                ),
                "chunk_key": self.generate_deterministic_chunk_key(
                    str(chunk["document_hash"]), int(chunk["chunk_index"])
                ),
            }
            for chunk in chunks
        ]

    def _vector_buffer(self, embedding: Any) -> bytes:
        if isinstance(embedding, memoryview):
            embedding = embedding.tobytes()
        if isinstance(embedding, (bytes, bytearray)):
            value = bytes(embedding)
            if len(value) != self.vector_dim * 4:
                raise ValueError("embedding 向量维度与 vector_dim 不一致。")
            return value
        try:
            values = [float(item) for item in embedding]
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding 返回值不是可用浮点向量。") from exc
        if len(values) != self.vector_dim or not all(math.isfinite(item) for item in values):
            raise ValueError("embedding 向量维度与 vector_dim 不一致。")
        buffer = array("f", values)
        if sys.byteorder != "little":
            buffer.byteswap()
        return buffer.tobytes()

    @staticmethod
    def _tag_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple, set)):
            return ",".join(str(item) for item in value)
        return str(value)

    @staticmethod
    def _validated_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in mapping.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Redis hash field 必须是非空字符串。")
            if value is None:
                normalized[key] = ""
            elif isinstance(value, bool):
                normalized[key] = "true" if value else "false"
            elif isinstance(value, (str, bytes, bytearray, int, float)):
                normalized[key] = bytes(value) if isinstance(value, bytearray) else value
            else:
                normalized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return normalized

    async def _prepare_index_documents(
        self,
        chunks: List[Dict[str, Any]],
        vectorizer: Any,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        prepared = self.prepare_chunks_for_replacement(chunks)
        contents = [str(chunk.get("content") or "") for chunk in prepared]
        if any(not content for content in contents):
            raise ValueError("空 chunk 不能写入 knowledge index。")
        embeddings = await vectorizer.aembed_many(contents)
        if len(embeddings) != len(prepared):
            raise ValueError("embedding 返回数量与 chunk 数量不一致。")

        created_at = datetime.now(timezone.utc).timestamp()
        documents: List[Dict[str, Any]] = []
        for chunk, embedding in zip(prepared, embeddings):
            content_hash = hashlib.sha256(
                str(chunk["content"]).encode("utf-8")
            ).hexdigest()
            chunk["content_hash"] = content_hash
            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            mapping = {
                "id": chunk["chunk_key"],
                "document_hash": str(chunk["document_hash"]),
                "content_hash": content_hash,
                "title": str(chunk.get("title") or ""),
                "content": str(chunk["content"]),
                "source": str(chunk.get("source") or ""),
                "category": str(chunk.get("category") or "shared"),
                "doc_type": str(chunk.get("doc_type") or "knowledge"),
                "name": str(chunk.get("name") or chunk.get("title") or ""),
                "summary": str(chunk.get("summary") or ""),
                "priority": str(chunk.get("priority") or "normal"),
                "pinned": str(chunk.get("pinned") or "false"),
                "severity": str(chunk.get("severity") or "medium"),
                "product_labels": self._tag_value(
                    chunk.get("product_labels") or metadata.get("product_labels")
                ),
                "product_label_tags": self._tag_value(
                    chunk.get("product_label_tags") or metadata.get("product_label_tags")
                ),
                "version": str(chunk.get("version") or "latest"),
                "chunk_index": int(chunk["chunk_index"]),
                "created_at": created_at,
                "vector": self._vector_buffer(embedding),
            }
            documents.append(self._validated_mapping(mapping))
        return prepared, documents

    async def _pipeline(self) -> Any:
        pipeline = self.client.pipeline(transaction=True)
        if inspect.isawaitable(pipeline):
            pipeline = await pipeline
        return pipeline

    async def _atomic_replace(
        self,
        *,
        prepared_chunks: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        old_document_hash: str,
        source_document_path: str,
    ) -> None:
        new_document_hash = str(prepared_chunks[0]["document_hash"])
        old_chunk_keys = (
            await self.find_existing_chunks(old_document_hash)
            if old_document_hash
            else []
        )
        if old_document_hash != new_document_hash:
            old_chunk_keys.extend(await self.find_existing_chunks(new_document_hash))
        old_chunk_keys = sorted(set(old_chunk_keys))

        first = prepared_chunks[0]
        timestamp = datetime.now(timezone.utc).isoformat()
        document_metadata = self._validated_mapping(
            {
                "document_hash": new_document_hash,
                "title": str(first.get("title") or ""),
                "source": str(first.get("source") or ""),
                "category": str(first.get("category") or "shared"),
                "doc_type": str(first.get("doc_type") or "knowledge"),
                "name": str(first.get("name") or first.get("title") or ""),
                "summary": str(first.get("summary") or ""),
                "priority": str(first.get("priority") or "normal"),
                "pinned": str(first.get("pinned") or "false"),
                "source_document_path": source_document_path,
                "source_document_scope": str(first.get("source_document_scope") or ""),
                "chunk_count": len(prepared_chunks),
                "total_content_length": sum(
                    len(str(chunk.get("content") or "")) for chunk in prepared_chunks
                ),
                "last_updated": timestamp,
            }
        )
        source_metadata = self._validated_mapping(
            {
                "document_hash": new_document_hash,
                "source_document_path": source_document_path,
                "source_document_scope": str(first.get("source_document_scope") or ""),
                "title": str(first.get("title") or ""),
                "source": str(first.get("source") or ""),
                "category": str(first.get("category") or "shared"),
                "doc_type": str(first.get("doc_type") or "knowledge"),
                "last_updated": timestamp,
            }
        )

        transaction = await self._pipeline()
        delete_keys = list(old_chunk_keys)
        if old_document_hash and old_document_hash != new_document_hash:
            delete_keys.append(self.generate_document_tracking_key(old_document_hash))
        if delete_keys:
            transaction.delete(*delete_keys)
        for chunk, document in zip(prepared_chunks, documents):
            transaction.hset(str(chunk["chunk_key"]), mapping=document)
        transaction.hset(
            self.generate_document_tracking_key(new_document_hash),
            mapping=document_metadata,
        )
        if source_document_path:
            transaction.hset(
                self.generate_source_tracking_key(source_document_path),
                mapping=source_metadata,
            )
        await transaction.execute()

    async def replace_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        vectorizer: Any,
    ) -> int:
        if not chunks:
            return 0
        document_hash = str(chunks[0]["document_hash"])
        expected_keys = {
            self.generate_deterministic_chunk_key(document_hash, int(chunk["chunk_index"]))
            for chunk in chunks
        }
        if set(await self.find_existing_chunks(document_hash)) == expected_keys:
            return 0
        prepared, documents = await self._prepare_index_documents(chunks, vectorizer)
        await self._atomic_replace(
            prepared_chunks=prepared,
            documents=documents,
            old_document_hash=document_hash,
            source_document_path="",
        )
        return len(documents)

    async def replace_source_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        vectorizer: Any,
    ) -> Dict[str, Any]:
        if not chunks:
            return {"action": "unchanged", "indexed_count": 0}
        source_path = str(chunks[0].get("source_document_path") or "").strip()
        if not source_path:
            count = await self.replace_document_chunks(chunks, vectorizer)
            return {"action": "add" if count else "unchanged", "indexed_count": count}

        document_hash = str(chunks[0]["document_hash"])
        tracked = await self.get_source_document_tracking(source_path)
        previous_hash = str(tracked.get("document_hash") or "") if tracked else ""
        expected_keys = {
            self.generate_deterministic_chunk_key(document_hash, int(chunk["chunk_index"]))
            for chunk in chunks
        }
        if previous_hash == document_hash:
            current_keys = set(await self.find_existing_chunks(document_hash))
            if current_keys == expected_keys:
                return {
                    "action": "unchanged",
                    "indexed_count": 0,
                    "document_hash": document_hash,
                    "previous_document_hash": previous_hash,
                    "source_document_path": source_path,
                }

        prepared, documents = await self._prepare_index_documents(chunks, vectorizer)
        await self._atomic_replace(
            prepared_chunks=prepared,
            documents=documents,
            old_document_hash=previous_hash,
            source_document_path=source_path,
        )
        return {
            "action": "update" if previous_hash else "add",
            "indexed_count": len(documents),
            "document_hash": document_hash,
            "previous_document_hash": previous_hash or None,
            "source_document_path": source_path,
        }
