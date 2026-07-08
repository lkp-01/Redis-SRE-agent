"""Agent 路由器。

原项目这里使用轻量 LLM 判断 REDIS_TRIAGE / REDIS_CHAT。阶段三禁止真实 OpenAI
调用，所以保留同名 `AgentType` 和 `route_to_appropriate_agent`，内部改成确定性规则。
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from redis_sre_agent.core.targets import get_attached_target_handles_from_context

logger = logging.getLogger(__name__)
BaseMessage = Any


class AgentType(Enum):
    """可用 Agent 类型，枚举值沿用原项目。"""

    REDIS_TRIAGE = "redis_triage"
    REDIS_CHAT = "redis_chat"
    KNOWLEDGE_ONLY = "knowledge_only"
    REDIS_FOCUSED = "redis_triage"

# 只裁剪聊天记录的最后四条
def format_conversation_context(
    conversation_history: Optional[List[BaseMessage]], max_messages: int = 4
) -> str:
    """把最近对话压成路由提示文本的轻量插槽。"""
    if not conversation_history:
        return ""
    recent = conversation_history[-max_messages:]
    return "\n".join(str(item)[:500] for item in recent if item is not None)


async def route_to_appropriate_agent(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    user_preferences: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[BaseMessage]] = None,
) -> AgentType:
    """用确定性 mock 规则保留原 router 接口，不调用 LLM。"""
    logger.info("Routing query with stage-three deterministic router.")

    context = context or {}
    has_instance = bool(context.get("instance_id"))
    has_cluster = bool(context.get("cluster_id"))
    has_attached_targets = bool(get_attached_target_handles_from_context(context))
    has_support_package = bool(context.get("support_package_path"))
    has_diagnostic_scope = bool(has_instance or has_cluster or has_attached_targets)

    if has_support_package:
        return AgentType.REDIS_TRIAGE

    preferred = (user_preferences or {}).get("preferred_agent")
    if has_diagnostic_scope and preferred in [agent.value for agent in AgentType]:
        return AgentType(preferred)

    normalized_query = str(query or "").lower()
    deep_markers = (
        "deep triage",
        "deep research",
        "deep analysis",
        "deep dive",
        "go deep",
        "comprehensive triage",
        "full triage",
        "exhaustive analysis",
        "thorough investigation",
    )
    if any(marker in normalized_query for marker in deep_markers):
        return AgentType.REDIS_TRIAGE
    return AgentType.REDIS_CHAT

#判断“需不需要去连接真实的 Redis 线上环境去查数据”的函数。只要出现了下面那些关键词就链接
async def query_needs_live_redis_scope(
    query: str,
    conversation_history: Optional[List[BaseMessage]] = None,
) -> bool:
    """阶段三确定性 live-scope 判断插槽。"""
    normalized_query = str(query or "").lower()
    live_markers = (
        "check",
        "diagnose",
        "triage",
        "slowlog",
        "memory",
        "clients",
        "connection",
        "redis",
    )
    return any(marker in normalized_query for marker in live_markers)
