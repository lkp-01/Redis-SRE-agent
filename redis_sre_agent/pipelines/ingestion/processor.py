"""摄取处理器的稳定公共导出。"""

from ._processor_impl import (
    IngestionPipeline,
    ensure_knowledge_index,
    get_knowledge_index,
    get_vectorizer,
)
from .document_processor import DocumentProcessor

__all__ = [
    "DocumentProcessor",
    "IngestionPipeline",
    "ensure_knowledge_index",
    "get_knowledge_index",
    "get_vectorizer",
]
