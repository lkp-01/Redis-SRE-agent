"""本地文档模型边界；网页 scraper 仍是后续插槽。"""

from .base import (
    ArtifactStorage,
    DocumentCategory,
    DocumentType,
    ScrapedDocument,
    SeverityLevel,
)

__all__ = [
    "ArtifactStorage",
    "DocumentCategory",
    "DocumentType",
    "ScrapedDocument",
    "SeverityLevel",
]
