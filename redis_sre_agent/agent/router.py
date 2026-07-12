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

from redis_sre_agent.core.config import settings
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
    # 遍历这些最近的消息，提取消息转成字符串，加入到lines
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
    # 没有大模型可以用是，使用基础路由判断规则
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
    #如果用户的查询里包含了上述任何一个触发词，就直接扔给深度排查 Agent
    if any(marker in normalized_query for marker in deep_markers):
        return AgentType.REDIS_TRIAGE
    #否则交给聊天处理
    return AgentType.REDIS_CHAT


def _get_optional_router_llm(
    context: Optional[Dict[str, Any]],
    user_preferences: Optional[Dict[str, Any]],
) -> Optional[Any]:
    # 从用户偏好中提取大模型实例
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
    # 获取上下文中附加的所有目标句柄（比如特定的节点或资源）
    attached_target_handles = get_attached_target_handles_from_context(context)
    # 判断是否有附加目标
    has_attached_targets = bool(attached_target_handles)
    # 判断是否上传了支持包（support package，通常用于离线排查诊断）
    has_support_package = bool(context.get("support_package_path"))

    # 综合判断：只要有实例、有集群，或者有附加目标中的任意一个，就认为当前具备“诊断范围 (diagnostic scope)”
    has_diagnostic_scope = bool(has_instance or has_cluster or has_attached_targets)

    # 规则 1：如果用户提供了 support package，说明是硬核排查，直接强制丢给 REDIS_TRIAGE（深度排查 Agent）
    if has_support_package:
        logger.info("Support package scope routes to REDIS_TRIAGE.")
        return AgentType.REDIS_TRIAGE

    # 规则 2：看看用户有没有明确指定想用哪个 Agent
    preferred = (user_preferences or {}).get("preferred_agent")
    # 如果用户有明确的诊断范围，且他们指定的 Agent 是系统支持的合法 Agent，就直接遵从用户的偏好
    if has_diagnostic_scope and preferred in [agent.value for agent in AgentType]:
        return AgentType(preferred)

    # 尝试从上下文或用户偏好中获取用于路由的 LLM 实例
    router_llm = _get_optional_router_llm(context, user_preferences)

    # 沿用 original 的 nano 模型路由边界。创建失败时仍可使用下面的确定性规则。
    if router_llm is None and settings.openai_api_key is not None:
        try:
            from redis_sre_agent.core.llm_helpers import create_nano_llm

            router_llm = create_nano_llm(timeout=10.0)
        except Exception as exc:
            logger.warning(
                "Router LLM initialization failed; using fallback route (error=%s).",
                type(exc).__name__,
            )

    # 如果成功获取到了路由专用的 LLM，就开始构造 Prompt 交给 LLM 判断
    if router_llm is not None:
        # 将历史对话压缩成字符串格式
        context_str = format_conversation_context(conversation_history)

        # 根据前面的状态，生成一句关于当前排查范围的简短提示词
        scope_hint = "none attached"
        if has_cluster and not has_instance:
            scope_hint = "cluster"
        elif has_instance:
            scope_hint = "instance"
        elif has_attached_targets:
            scope_hint = f"{len(attached_target_handles)} attached targets"

        # 设定系统级 Prompt，严格限制大模型只能回答 DEEP_TRIAGE 或 CHAT
        prompt = """You route requests for a Redis SRE agent.

Respond with exactly one token:
- DEEP_TRIAGE for explicit deep, exhaustive, comprehensive triage requests.
- CHAT for all other Redis questions, including normal health checks."""
        try:
            # 调用封装好的受保护的大模型请求方法
            response = await guarded_ainvoke(
                router_llm,
                [
                    SystemMessage(content=prompt),  # 系统设定
                    # 把当前范围、用户问题和历史上下文喂给模型
                    HumanMessage(content=f"Scope: {scope_hint}\nQuery: {query}{context_str}"),
                ],
                request_kind="router.route_to_appropriate_agent",
            )

            # 清理模型的回答（转大写并去空格）
            category = str(getattr(response, "content", response)).strip().upper()

            # 根据模型回复的关键字进行精确分发
            if "DEEP_TRIAGE" in category:
                return AgentType.REDIS_TRIAGE
            if "CHAT" in category:
                return AgentType.REDIS_CHAT

        except Exception as exc:
            # 如果大模型接口挂了、超时了，或者返回了乱码，记录警告日志，然后继续往下走（去触发降级机制）
            logger.warning(
                "Router LLM failed; using fallback route (error=%s).",
                type(exc).__name__,
            )

    # 降级路线：如果没有 LLM 或者 LLM 崩溃了，用纯文本正则匹配来兜底判断
    return _fallback_route(
        query,
        has_cluster=has_cluster,
        has_instance=has_instance,
        has_attached_targets=has_attached_targets,
    )

# 判断没有附带任何环境的查询，是不是需要链接真实redis
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
