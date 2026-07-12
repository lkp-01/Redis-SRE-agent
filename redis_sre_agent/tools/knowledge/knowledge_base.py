"""只读 knowledge search ToolProvider。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from redis_sre_agent.core.knowledge_helpers import (
    _coerce_non_negative_int,
    _coerce_positive_int,
    search_knowledge_base_helper,
)
from redis_sre_agent.tools.models import ToolCapability, ToolDefinition
from redis_sre_agent.tools.protocols import ToolProvider


class KnowledgeBaseToolProvider(ToolProvider):
    """只向 LLM 暴露一个 READ search；摄取只能从显式 pipeline 入口进行。"""

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
                    "搜索 Redis SRE knowledge base 中的 runbook、诊断说明和操作文档。"
                    "使用结果时必须保留 title 与 source 引用。"
                ),
                capability=ToolCapability.KNOWLEDGE,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要查找的 Redis 诊断知识。",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "offset": {
                            "type": "integer",
                            "default": 0,
                            "minimum": 0,
                        },
                        "version": {
                            "type": "string",
                            "default": "latest",
                        },
                        "distance_threshold": {
                            "type": "number",
                            "description": "可选 cosine distance 上限；null 表示纯 KNN。",
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
        return await search_knowledge_base_helper(
            query=query,
            limit=min(_coerce_positive_int(limit, default=10), 50),
            offset=_coerce_non_negative_int(offset, default=0),
            version=version,
            distance_threshold=distance_threshold,
        )
