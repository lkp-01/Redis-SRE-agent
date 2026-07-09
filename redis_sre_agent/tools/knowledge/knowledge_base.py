"""Stage 5 dummy knowledge provider。

原项目的 KnowledgeBaseToolProvider 暴露多个 RAG、skill 和 support ticket 工具。当前阶段
只保留 `search` 的同名入口，返回空结果，避免提前实现 Stage 8 的知识库流水线。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from redis_sre_agent.tools.models import ToolCapability, ToolDefinition
from redis_sre_agent.tools.protocols import ToolProvider


class KnowledgeBaseToolProvider(ToolProvider):
    """只提供 dummy search 的 knowledge provider。"""

    @property
    def provider_name(self) -> str:
        return "knowledge"

    @property
    def requires_redis_instance(self) -> bool:
        return False

    def create_tool_schemas(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name=self._make_tool_name("search"),
                description=(
                    "Search the Redis SRE knowledge base. In this Stage 5 reproduction this "
                    "is a dummy RAG slot and returns no documents; real ingestion, embedding, "
                    "and vector retrieval are future-stage work."
                ),
                capability=ToolCapability.KNOWLEDGE,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query describing the needed Redis knowledge.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return.",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Number of results to skip.",
                            "default": 0,
                            "minimum": 0,
                        },
                        "version": {
                            "type": "string",
                            "description": "Documentation version filter placeholder.",
                            "default": "latest",
                        },
                        "distance_threshold": {
                            "type": "number",
                            "description": "Semantic distance threshold placeholder.",
                        },
                    },
                    "required": ["query"],
                },
            )
        ]

    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        version: Optional[str] = "latest",
        distance_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """返回空知识结果，明确告诉上层真实 RAG 尚未启用。"""
        return {
            "status": "success",
            "query": query,
            "limit": limit,
            "offset": offset,
            "version": version,
            "distance_threshold": distance_threshold,
            "retrieval_kind": "dummy_knowledge",
            "retrieval_label": "Dummy knowledge slot",
            "results": [],
            "message": "Stage 5 仅保留知识库工具插槽，真实 RAG 检索尚未启用。",
        }
