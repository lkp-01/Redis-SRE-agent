"""Knowledge 工具插槽。

Stage 5 只需要一个可加载的 dummy search 工具，让 Agent 主链路保持原项目的 RAG
接缝；真实知识库摄取、embedding 和向量检索留到后续阶段。
"""

from .knowledge_base import KnowledgeBaseToolProvider

__all__ = ["KnowledgeBaseToolProvider"]
