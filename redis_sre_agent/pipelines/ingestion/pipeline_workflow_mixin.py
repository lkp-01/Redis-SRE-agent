"""只处理本地 Markdown 与 artifact batch 的高层工作流。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .processor_source_helpers import (
    create_scraped_document_from_markdown,
    find_markdown_files,
)


class PipelineWorkflowMixin:
    """恢复 original 方法名，但裁掉网页、包扩展和多索引能力。"""

    @staticmethod
    def _load_source_markdown_files(source_dir: Path, *, action: str) -> List[Path]:
        del action
        if not source_dir.exists() or not source_dir.is_dir():
            raise ValueError("source directory 不存在或不是目录。")
        return find_markdown_files(source_dir)

    def _load_source_documents(self, source_dir: Path, *, action: str) -> List[Any]:
        return [
            create_scraped_document_from_markdown(path, source_dir)
            for path in self._load_source_markdown_files(source_dir, action=action)
        ]

    async def prepare_source_artifacts(self, source_dir: Path, batch_date: str) -> int:
        if self.storage is None:
            raise ValueError("prepare 需要 ArtifactStorage。")
        self.storage.set_batch_date(batch_date)
        documents = self._load_source_documents(Path(source_dir), action="prepare")
        for document in documents:
            self.storage.save_document(document)
        if documents:
            self.storage.save_batch_manifest()
        return len(documents)

    async def ingest_prepared_batch(self, batch_date: str) -> List[Dict[str, Any]]:
        result = await self.ingest_batch(batch_date)
        if result.get("success"):
            return [{"status": "success", "batch_date": batch_date, **result}]
        return [
            {
                "status": "error",
                "batch_date": batch_date,
                "error": "prepared batch 摄取失败。",
                **result,
            }
        ]

    async def list_ingested_batches(self) -> List[Dict[str, Any]]:
        if self.storage is None:
            return []
        batches: List[Dict[str, Any]] = []
        for batch_date in self.storage.list_available_batches():
            item: Dict[str, Any] = {"batch_date": batch_date, "ingested": False}
            manifest = self.storage.base_path / batch_date / "ingestion_manifest.json"
            if manifest.exists():
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    item.update(payload)
                    item["ingested"] = bool(payload.get("success"))
            batches.append(item)
        return sorted(batches, key=lambda item: item["batch_date"], reverse=True)

    async def reindex_batch(self, batch_date: str) -> Dict[str, Any]:
        return await self.ingest_batch(batch_date)
