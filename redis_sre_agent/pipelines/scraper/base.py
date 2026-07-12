"""摄取文档的基础枚举和可序列化模型。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DocumentCategory(str, Enum):
    OSS = "oss"
    ENTERPRISE = "enterprise"
    SHARED = "shared"


class DocumentType(str, Enum):
    RUNBOOK = "runbook"
    DOCUMENTATION = "documentation"
    BLOG_POST = "blog_post"
    TUTORIAL = "tutorial"
    TROUBLESHOOTING = "troubleshooting"
    REFERENCE = "reference"
    API_DOC = "api_doc"
    KNOWLEDGE = "knowledge"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScrapedDocument:
    """original 兼容的摄取文档；hash 不依赖采集时间。"""

    def __init__(
        self,
        title: str,
        content: str,
        source_url: str,
        category: DocumentCategory,
        doc_type: DocumentType,
        severity: SeverityLevel = SeverityLevel.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.title = title
        self.content = content
        self.source_url = source_url
        self.category = category
        self.doc_type = doc_type
        self.severity = severity
        self.metadata = metadata or {}
        self.scraped_at = datetime.now(timezone.utc)
        self.content_hash = self._generate_content_hash()

    @property
    def document_hash(self) -> str:
        """knowledge schema 使用的稳定文档 hash。"""

        return self.content_hash

    def _generate_content_hash(self) -> str:
        content_identity = f"{self.title}||{self.content}||{self.source_url}"
        return hashlib.sha256(content_identity.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "source_url": self.source_url,
            "category": self.category.value,
            "doc_type": self.doc_type.value,
            "severity": self.severity.value,
            "metadata": self.metadata,
            "scraped_at": self.scraped_at.isoformat(),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScrapedDocument":
        document = cls(
            title=str(data["title"]),
            content=str(data["content"]),
            source_url=str(data["source_url"]),
            category=DocumentCategory(data["category"]),
            doc_type=DocumentType(data["doc_type"]),
            severity=SeverityLevel(data.get("severity", SeverityLevel.MEDIUM.value)),
            metadata=dict(data.get("metadata") or {}),
        )
        if data.get("scraped_at"):
            document.scraped_at = datetime.fromisoformat(
                str(data["scraped_at"]).replace("Z", "+00:00")
            )
        # 重新计算而不是信任 artifact 中可被篡改的 hash。
        document.content_hash = document._generate_content_hash()
        return document


class ArtifactStorage:
    """按日期保存本地 JSON artifact 和 batch/ingestion manifest。"""

    def __init__(self, base_path: Union[str, Path], create_dirs: bool = False) -> None:
        self.base_path = Path(base_path)
        self.current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.current_batch_path = self.base_path / self.current_date
        self._dirs_created = False
        if create_dirs:
            self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        if self._dirs_created:
            return
        self.current_batch_path.mkdir(parents=True, exist_ok=True)
        for category in DocumentCategory:
            (self.current_batch_path / category.value).mkdir(exist_ok=True)
        self._dirs_created = True

    def set_batch_date(self, batch_date: str) -> None:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", batch_date):
            raise ValueError("batch date 必须使用 YYYY-MM-DD 格式。")
        # datetime 校验月份和日期是否真实存在。
        datetime.strptime(batch_date, "%Y-%m-%d")
        self.current_date = batch_date
        self.current_batch_path = self.base_path / batch_date
        self._dirs_created = False

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
        """先写同目录临时文件再替换，避免留下半个 artifact。"""

        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def save_document(self, document: ScrapedDocument) -> Path:
        self._ensure_dirs()
        category_path = self.current_batch_path / document.category.value
        safe_title = "".join(
            character
            for character in document.title
            if character.isalnum() or character in (" ", "-", "_")
        ).rstrip()
        safe_title = (safe_title.replace(" ", "_")[:50] or "document")
        path = category_path / f"{safe_title}_{document.content_hash}.json"
        return self._write_json(path, document.to_dict())

    def save_batch_manifest(self) -> Path:
        self._ensure_dirs()
        documents: List[Dict[str, Any]] = []
        for path in sorted(self.current_batch_path.rglob("*.json")):
            if path.name in {"batch_manifest.json", "ingestion_manifest.json"}:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(payload, dict):
                documents.append(payload)
        manifest: Dict[str, Any] = {
            "batch_date": self.current_date,
            "total_documents": len(documents),
            "categories": {},
            "document_types": {},
            "sources": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        for document in documents:
            category = str(document.get("category") or "unknown")
            doc_type = str(document.get("doc_type") or "unknown")
            manifest["categories"][category] = manifest["categories"].get(category, 0) + 1
            manifest["document_types"][doc_type] = (
                manifest["document_types"].get(doc_type, 0) + 1
            )
            source = str(document.get("source_url") or document.get("source") or "").strip()
            if source and source not in manifest["sources"]:
                manifest["sources"].append(source)
        return self._write_json(self.current_batch_path / "batch_manifest.json", manifest)

    def save_ingestion_manifest(self, batch_date: str, payload: Dict[str, Any]) -> Path:
        path = self.base_path / batch_date / "ingestion_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return self._write_json(path, payload)

    def list_available_batches(self) -> List[str]:
        if not self.base_path.exists():
            return []
        return sorted(
            item.name
            for item in self.base_path.iterdir()
            if item.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", item.name)
        )

    def get_batch_manifest(self, batch_date: str) -> Optional[Dict[str, Any]]:
        path = self.base_path / batch_date / "batch_manifest.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
