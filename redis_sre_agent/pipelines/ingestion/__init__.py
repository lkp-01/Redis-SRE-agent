"""本地 Markdown 到 knowledge vector index 的摄取边界。"""

from .document_processor import DocumentProcessor
from .processor import IngestionPipeline

__all__ = ["DocumentProcessor", "IngestionPipeline"]
