"""本地 Markdown 解析和稳定来源身份 helper。"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from redis_sre_agent.pipelines.scraper.base import (
    DocumentCategory,
    DocumentType,
    ScrapedDocument,
    SeverityLevel,
)

logger = logging.getLogger(__name__)

CATEGORY_NAME_MAP = {
    "oss": DocumentCategory.OSS,
    "enterprise": DocumentCategory.ENTERPRISE,
    "shared": DocumentCategory.SHARED,
    "cloud": DocumentCategory.SHARED,
}
SEVERITY_NAME_MAP = {
    "critical": SeverityLevel.CRITICAL,
    "high": SeverityLevel.HIGH,
    "warning": SeverityLevel.MEDIUM,
    "medium": SeverityLevel.MEDIUM,
    "normal": SeverityLevel.MEDIUM,
    "low": SeverityLevel.LOW,
    "info": SeverityLevel.LOW,
}
RESERVED_METADATA_KEYS = {
    "file_path",
    "file_size",
    "original_category",
    "original_severity",
    "original_doc_type",
    "determined_category",
    "doc_type",
    "name",
    "summary",
    "priority",
    "pinned",
    "source_document_path",
    "source_document_scope",
}


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    return default


def strip_yaml_front_matter(text: str) -> tuple[str, bool]:
    if not text.startswith("---"):
        return text, False
    match = re.match(r"^---\s*\n.*?\n---\s*(?:\n|$)", text, re.DOTALL)
    if not match:
        return text, False
    return text[match.end() :], True


def normalize_metadata_key(key: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", key.strip().lower())
    return re.sub(r"[^\w]", "", normalized)


def parse_markdown_metadata(content: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    front_matter = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", content, re.DOTALL)
    if front_matter:
        for line in front_matter.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[normalize_metadata_key(key)] = value.strip().strip('"').strip("'")
    title_match = re.search(r"^# (.+)", content, re.MULTILINE)
    if title_match and "title" not in metadata:
        metadata["title"] = title_match.group(1).strip()
    for match in re.finditer(r"^\*\*([^*]+)\*\*:\s*(.+)$", content, re.MULTILINE):
        metadata.setdefault(normalize_metadata_key(match.group(1)), match.group(2).strip())
    return metadata


def normalize_doc_type(value: str) -> tuple[DocumentType, str]:
    normalized = re.sub(r"[\s-]+", "_", str(value or "knowledge").strip().lower())
    try:
        return DocumentType(normalized), normalized
    except ValueError:
        logger.debug("未知 doc_type，按 knowledge 处理。")
        return DocumentType.KNOWLEDGE, "knowledge"


def normalize_priority(value: Any) -> str:
    normalized = str(value or "normal").strip().lower()
    return normalized if normalized in {"low", "normal", "high", "critical"} else "normal"


def find_source_documents_root(source_dir: Path) -> Path:
    resolved = source_dir.resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "source_documents":
            return candidate
    return resolved


def resolve_source_document_identity(md_file: Path, source_dir: Path) -> tuple[str, str]:
    resolved_file = md_file.resolve()
    resolved_source = source_dir.resolve()
    source_root = find_source_documents_root(source_dir)
    try:
        relative_path = resolved_file.relative_to(source_root).as_posix()
    except ValueError:
        relative_path = resolved_file.relative_to(resolved_source).as_posix()
    try:
        scope = resolved_source.relative_to(source_root).as_posix()
    except ValueError:
        scope = ""
    if scope in {"", "."}:
        return relative_path, ""
    return relative_path, f"{scope.rstrip('/')}/"


def determine_document_category(path: Path, metadata: Dict[str, Any]) -> DocumentCategory:
    explicit = str(metadata.get("category") or "").strip().lower()
    if explicit in CATEGORY_NAME_MAP:
        return CATEGORY_NAME_MAP[explicit]
    for part in path.parts:
        normalized = part.lower()
        if normalized in CATEGORY_NAME_MAP:
            return CATEGORY_NAME_MAP[normalized]
    return DocumentCategory.SHARED


def create_scraped_document_from_markdown(
    md_file: Path,
    source_dir: Optional[Path] = None,
) -> ScrapedDocument:
    content = md_file.read_text(encoding="utf-8")
    metadata = parse_markdown_metadata(content)
    source_path = ""
    source_scope = ""
    if source_dir is not None:
        source_path, source_scope = resolve_source_document_identity(md_file, source_dir)

    title = metadata.get("title", md_file.stem.replace("-", " ").title())
    category = determine_document_category(md_file, metadata)
    priority = normalize_priority(metadata.get("priority"))
    severity_raw = str(metadata.get("severity") or priority).strip().lower()
    severity = SEVERITY_NAME_MAP.get(severity_raw, SeverityLevel.MEDIUM)
    doc_type_raw = str(metadata.get("doc_type") or "knowledge")
    doc_type, normalized_doc_type = normalize_doc_type(doc_type_raw)
    name = str(metadata.get("name") or md_file.stem).strip() or md_file.stem
    explicit_url = str(metadata.get("url") or "").strip()
    stable_local_url = f"file://{source_path}" if source_path else f"file://{md_file.name}"
    passthrough = {
        key: value for key, value in metadata.items() if key not in RESERVED_METADATA_KEYS
    }
    return ScrapedDocument(
        title=title,
        content=content,
        source_url=explicit_url or stable_local_url,
        category=category,
        doc_type=doc_type,
        severity=severity,
        metadata={
            **passthrough,
            "file_path": source_path or md_file.name,
            "file_size": md_file.stat().st_size,
            "original_category": str(metadata.get("category") or "shared").lower(),
            "original_severity": severity_raw,
            "original_doc_type": doc_type_raw,
            "determined_category": category.value,
            "doc_type": normalized_doc_type,
            "name": name,
            "summary": str(metadata.get("summary") or "").strip() or None,
            "priority": priority,
            "pinned": parse_bool(metadata.get("pinned"), default=False),
            "source_document_path": source_path,
            "source_document_scope": source_scope,
        },
    )


def find_markdown_files(source_dir: Path) -> List[Path]:
    return sorted(
        (path for path in source_dir.rglob("*.md") if path.name.lower() != "readme.md"),
        key=lambda path: path.as_posix(),
    )
