"""Agent 路由逻辑。

original 的 router 用轻量 LLM 判断 REDIS_CHAT / REDIS_TRIAGE。裁剪版保留
这个结构：如果调用方注入 router_llm，就先走 LLM 分类；没有真实 LLM 时才落到
本地 fallback，避免测试环境触发 OpenAI API。
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from redis_sre_agent.core.targets import get_attached_target_handles_from_context

from .helpers import guarded_ainvoke

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """可用 Agent 类型，枚举值沿用 original。"""

    REDIS_TRIAGE = "redis_triage"
    REDIS_CHAT = "redis_chat"
    KNOWLEDGE_ONLY = "knowledge_only"
    REDIS_FOCUSED = "redis_triage"


def format_conversation_context(
    conversation_history: Optional[List[BaseMessage]],
    max_messages: int = 4,
) -> str:
    """把最近对话压成 router 分类提示。"""

    if not conversation_history:
        return ""
    recent = conversation_history[-max_messages:]
    lines = ["\n\nRecent conversation context:"]
    for message in recent:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        content = str(getattr(message, "content", message))
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _fallback_route(
    query: str,
    *,
    has_cluster: bool,
    has_instance: bool,
    has_attached_targets: bool,
) -> AgentType:
    normalized_query = str(query or "").lower()
    deep_markers = (
        "deep triage",
        "deep research",
        "deep analysis",
        "deep dive",
        "go deep",
        "comprehensive triage",
        "comprehensive analysis",
        "full triage",
        "exhaustive",
        "thorough investigation",
    )
    if any(marker in normalized_query for marker in deep_markers):
        return AgentType.REDIS_TRIAGE
    return AgentType.REDIS_CHAT


def _get_optional_router_llm(
    context: Optional[Dict[str, Any]],
    user_preferences: Optional[Dict[str, Any]],
) -> Optional[Any]:
    for source in (user_preferences, context):
        if isinstance(source, dict) and source.get("router_llm") is not None:
            return source["router_llm"]
    return None


async def route_to_appropriate_agent(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    user_preferences: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[BaseMessage]] = None,
) -> AgentType:
    """选择 ChatAgent 或 SRELangGraphAgent。"""

    context = context or {}
    has_instance = bool(context.get("instance_id"))
    has_cluster = bool(context.get("cluster_id"))
    attached_target_handles = get_attached_target_handles_from_context(context)
    has_attached_targets = bool(attached_target_handles)
    has_support_package = bool(context.get("support_package_path"))
    has_diagnostic_scope = bool(has_instance or has_cluster or has_attached_targets)

    if has_support_package:
        logger.info("Support package scope routes to REDIS_TRIAGE.")
        return AgentType.REDIS_TRIAGE

    preferred = (user_preferences or {}).get("preferred_agent")
    if has_diagnostic_scope and preferred in [agent.value for agent in AgentType]:
        return AgentType(preferred)

    router_llm = _get_optional_router_llm(context, user_preferences)
    if router_llm is not None:
        context_str = format_conversation_context(conversation_history)
        scope_hint = "none attached"
        if has_cluster and not has_instance:
            scope_hint = "cluster"
        elif has_instance:
            scope_hint = "instance"
        elif has_attached_targets:
            scope_hint = f"{len(attached_target_handles)} attached targets"
        prompt = """You route requests for a Redis SRE agent.

Respond with exactly one token:
- DEEP_TRIAGE for explicit deep, exhaustive, comprehensive triage requests.
- CHAT for all other Redis questions, including normal health checks."""
        try:
            response = await guarded_ainvoke(
                router_llm,
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"Scope: {scope_hint}\nQuery: {query}{context_str}"),
                ],
                request_kind="router.route_to_appropriate_agent",
            )
            category = str(getattr(response, "content", response)).strip().upper()
            if "DEEP_TRIAGE" in category:
                return AgentType.REDIS_TRIAGE
            if "CHAT" in category:
                return AgentType.REDIS_CHAT
        except Exception as exc:
            logger.warning("Router LLM failed; using fallback route: %s", exc)

    return _fallback_route(
        query,
        has_cluster=has_cluster,
        has_instance=has_instance,
        has_attached_targets=has_attached_targets,
    )


async def query_needs_live_redis_scope(
    query: str,
    conversation_history: Optional[List[BaseMessage]] = None,
) -> bool:
    """判断零 scope 查询是否需要 Redis live 工具。"""

    normalized_query = str(query or "").lower()
    diagnostic_markers = (
        "check",
        "diagnose",
        "triage",
        "slowlog",
        "memory",
        "client",
        "connection",
        "latency",
        "replication",
        "cluster",
        "config",
        "info",
        "health",
        "issue",
        "problem",
    )
    knowledge_markers = (
        "what is",
        "how does",
        "best practice",
        "explain",
        "documentation",
        "general",
    )
    if any(marker in normalized_query for marker in diagnostic_markers):
        return True
    if "redis" in normalized_query and any(marker in normalized_query for marker in knowledge_markers):
        return False
    return "redis" in normalized_query
