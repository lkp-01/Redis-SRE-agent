"""
AI 把人类语言翻译成真实的服务器编号，并顺手把这台服务器需要的专用工具全部准备好
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from redis_sre_agent.core.targets import (
    bind_target_matches,
    list_known_targets,
    resolve_target_query,
)
from redis_sre_agent.targets.contracts import DiscoveryResponse
from redis_sre_agent.tools.models import ToolCapability, ToolDefinition
from redis_sre_agent.tools.protocols import ToolProvider


class TargetDiscoveryToolProvider(ToolProvider):
    """把自然语言 target 描述解析成安全 Redis target handle。"""

    def __init__(self, redis_instance=None, config=None):
        super().__init__(redis_instance=redis_instance, config=config)
        self._instance_hash = hashlib.sha256(self.provider_name.encode()).hexdigest()[:6]

    @property
    def provider_name(self) -> str:
        return "target_discovery"

    @property
    def requires_redis_instance(self) -> bool:
        return False

    def create_tool_schemas(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name=self._make_tool_name("list_known_redis_targets"),
                description=(
                    "List safe Redis targets currently known in the target catalog. "
                    "This returns public metadata only and does not attach live tools."
                ),
                capability=ToolCapability.UTILITIES,
                parameters={
                    "type": "object",
                    "properties": {
                        "target_kind": {"type": "string", "enum": ["instance", "cluster"]},
                        "environment": {"type": "string"},
                        "capability": {"type": "string"},
                        "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                        "offset": {"type": "integer", "default": 0, "minimum": 0},
                        "include_aliases": {"type": "boolean", "default": False},
                    },
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("resolve_redis_targets"),
                description=(
                    "Resolve natural-language Redis target descriptions into secret-safe matches "
                    "and opaque target handles."
                ),
                capability=ToolCapability.UTILITIES,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "allow_multiple": {"type": "boolean", "default": False},
                        "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
                        "attach_tools": {"type": "boolean", "default": True},
                        "preferred_capabilities": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]

    async def list_known_redis_targets(
        self,
        target_kind: Optional[str] = None,
        environment: Optional[str] = None,
        capability: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        include_aliases: bool = False,
    ) -> Dict[str, Any]:
        manager = getattr(self, "_manager", None)
        user_id = getattr(manager, "user_id", None)
        toolset_generation = manager.get_toolset_generation() if manager else 0
        payload = await list_known_targets(
            user_id=user_id,
            target_kind=target_kind,
            environment=environment,
            capability=capability,
            limit=limit,
            offset=offset,
            include_aliases=include_aliases,
        )
        payload["toolset_generation"] = toolset_generation
        return payload

    async def resolve_redis_targets(
        self,
        query: str,
        allow_multiple: bool = False,
        max_results: int = 5,
        attach_tools: bool = True,
        preferred_capabilities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        manager = getattr(self, "_manager", None)
        thread_id = getattr(manager, "thread_id", None)
        task_id = getattr(manager, "task_id", None)
        user_id = getattr(manager, "user_id", None)

        result = await resolve_target_query(
            query=query,
            user_id=user_id,
            allow_multiple=allow_multiple,
            max_results=max_results,
            preferred_capabilities=preferred_capabilities,
        )

        attached_handles: List[str] = []
        scope = None
        toolset_generation = manager.get_toolset_generation() if manager else 0
        if attach_tools and manager and result.status == "resolved" and result.selected_matches:
            scope = await bind_target_matches(
                matches=result.selected_matches,
                thread_id=thread_id,
                task_id=task_id,
                replace_existing=not allow_multiple,
                manager=manager,
            )
            attached_handles = scope.context_updates.get("attached_target_handles", [])
            toolset_generation = scope.toolset_generation

        payload = (
            result.public_dump() if isinstance(result, DiscoveryResponse) else result.model_dump()
        )
        payload["attached_target_handles"] = attached_handles
        payload["toolset_generation"] = toolset_generation
        if scope is not None:
            payload.update(
                {
                    key: value
                    for key, value in scope.context_updates.items()
                    if key not in {"instance_id", "cluster_id"}
                }
            )
        return payload
