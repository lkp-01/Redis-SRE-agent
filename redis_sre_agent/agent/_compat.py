"""Agent 测试 fallback。

正式 Agent workflow 使用真实 LangGraph / LangChain runtime。本文件只保留 fake
tool-calling LLM，供测试或未配置外部 LLM 的本地运行验证工具循环，不提供模拟
StateGraph 或模拟消息对象。
"""

from __future__ import annotations

import json
import re
from copy import copy
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .helpers import coerce_response_text, extract_tool_operation_name


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return coerce_response_text(value)


def _last_user_query(messages: Sequence[Any]) -> str:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            text = _message_text(message.content)
            matches = re.findall(r"User Query:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
            if matches:
                return matches[-1].strip()
            return text.strip()
    return ""


def _last_target_hint(messages: Sequence[Any]) -> str:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            text = _message_text(message.content)
            match = re.search(
                r"Target Hint:\s*(.+?)(?:\n\s*\n|$)",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                return match.group(1).strip()
    return ""


def _parse_tool_message_payload(message: ToolMessage) -> Dict[str, Any]:
    content = _message_text(message.content)
    if not content:
        return {}
    try:
        first_brace = content.find("{")
        payload = content[first_brace:] if first_brace >= 0 else content
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
        return {"raw": parsed}
    except Exception:
        return {"raw": content[:4000]}


def _envelopes_from_tool_messages(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    envelopes: List[Dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, ToolMessage):
            continue
        tool_name = str(message.name or "tool")
        data = _parse_tool_message_payload(message)
        status = str(data.get("status", "")).lower()
        envelopes.append(
            {
                "tool_key": tool_name,
                "name": extract_tool_operation_name(tool_name),
                "args": {},
                "status": "error" if status in {"error", "failed", "failure"} else "success",
                "data": data,
            }
        )
    return envelopes


def _operation_signature(name: str, data: Optional[Dict[str, Any]] = None) -> str:
    operation = extract_tool_operation_name(name)
    data = data or {}
    if operation == "info":
        return f"info:{data.get('section') or 'all'}"
    if operation == "config_get":
        return f"config_get:{data.get('pattern') or data.get('args', {}).get('pattern') or '*'}"
    if operation == "search_index_info":
        return f"search_index_info:{data.get('index_name') or ''}"
    return operation


def _tool_name_by_operation(tools: Sequence[Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for tool in tools or []:
        name = str(getattr(tool, "name", "") or "")
        if not name:
            continue
        result.setdefault(extract_tool_operation_name(name), name)
    return result


def _tool_call(tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "id": f"call_{uuid4().hex[:10]}",
        "name": tool_name,
        "args": dict(args or {}),
    }


def _fake_final_text(query: str, agent_kind: str) -> str:
    """测试 LLM 的可识别终态文本，不承担正式诊断推理。"""

    if agent_kind == "triage":
        return (
            "## Initial Assessment\nFake triage assessment for local tests.\n\n"
            "## What I'm Seeing\nTool evidence was collected.\n\n"
            "## My Recommendation\nReview the captured evidence.\n\n"
            "## Supporting Info\nLocal fake LLM output."
        )
    return (
        f"# Redis 诊断摘要\n\n查询：{query}\n\n"
        "## Evidence 摘要\n工具 evidence 已收集。\n\n"
        "## 下一步建议\n请根据 evidence 继续分析。"
    )


class _FakeStructuredLLM:
    def __init__(self, parent: "FakeToolCallingLLM", schema: Any) -> None:
        self.parent = parent
        self.schema = schema

    async def ainvoke(self, messages: Sequence[Any]) -> Any:
        from .models import (
            Recommendation,
            RecommendationStep,
            TargetSelectionDecision,
            Topic,
            TopicsList,
        )

        if self.schema is TargetSelectionDecision:
            import json

            from .router import query_needs_live_redis_scope

            raw_payload = _message_text(getattr(messages[-1], "content", "")) if messages else ""
            try:
                payload = json.loads(raw_payload)
            except (TypeError, ValueError):
                payload = {}
            query = str(payload.get("query") or "")
            targets = [
                item for item in payload.get("targets") or [] if isinstance(item, dict)
            ]
            requires_live = await query_needs_live_redis_scope(query)
            selected_target = None
            if requires_live and targets:
                selected_target = str(targets[0].get("display_name") or "").strip() or None
            return TargetSelectionDecision(
                requires_live_diagnostics=requires_live,
                selected_target=selected_target,
                reason_code="local_fake_fallback",
                confidence=0.5,
            )

        if self.schema is TopicsList:
            payload = "\n".join(_message_text(getattr(message, "content", "")) for message in messages)
            keys = re.findall(r'"tool_key"\s*:\s*"([^"]+)"', payload)
            return TopicsList(
                items=[
                    Topic(
                        id="T1",
                        title="Redis diagnostic evidence",
                        category="Performance",
                        severity="medium",
                        narrative="Review the captured Redis signals.",
                        evidence_keys=keys[:3],
                    )
                ]
            )
        if self.schema is Recommendation:
            return Recommendation(
                topic_id="T1",
                title="Redis diagnostic evidence",
                steps=[RecommendationStep(description="Review the captured tool evidence.")],
                verification=["Confirm the relevant Redis metrics again."],
            )
        raise TypeError(f"Unsupported fake structured schema: {self.schema}")


def _diagnostic_goals(agent_kind: str) -> List[tuple[str, Dict[str, Any]]]:
    goals: List[tuple[str, Dict[str, Any]]] = [
        ("info", {"section": "memory"}),
        ("info", {"section": "stats"}),
        ("client_list", {}),
        ("slowlog", {"count": 5}),
        ("memory_stats", {}),
        ("config_get", {"pattern": "maxmemory*"}),
        ("replication_info", {}),
    ]
    if agent_kind == "triage":
        goals.extend(
            [
                ("info", {"section": "clients"}),
                ("info", {"section": "keyspace"}),
                ("acl_log", {"count": 5}),
                ("cluster_info", {}),
                ("search_indexes", {}),
            ]
        )
    return goals


class FakeToolCallingLLM:
    """测试用 tool-calling LLM。

    它只负责返回 LangChain `AIMessage(tool_calls=...)`；工具执行仍必须经过
    Agent tool node 和 ToolManager。
    """

    max_tool_calls_per_turn = 4

    def __init__(self, *, agent_kind: str):
        self.agent_kind = agent_kind
        self._tools: List[Any] = []

    def bind_tools(self, tools: Sequence[Any]) -> "FakeToolCallingLLM":
        bound = copy(self)
        bound._tools = list(tools or [])
        return bound

    def with_structured_output(self, schema: Any, **_: Any) -> _FakeStructuredLLM:
        return _FakeStructuredLLM(self, schema)

    async def ainvoke(self, messages: Sequence[Any]) -> AIMessage:
        system_text = "\n".join(
            _message_text(getattr(message, "content", ""))
            for message in messages
            if getattr(message, "type", "") == "system"
        )
        if "careful technical editor" in system_text:
            return AIMessage(content=_fake_final_text(_last_user_query(messages), "triage"))
        if "research and then synthesize recommendations" in system_text:
            return AIMessage(content="Evidence is sufficient for structured synthesis.")
        if "iteration budget" in system_text or "final Redis triage response" in system_text:
            return AIMessage(content=_fake_final_text(_last_user_query(messages), self.agent_kind))

        query = _last_user_query(messages)
        target_hint = _last_target_hint(messages)
        envelopes = _envelopes_from_tool_messages(messages)
        operation_to_name = _tool_name_by_operation(self._tools)
        collected = {
            _operation_signature(str(env.get("tool_key") or env.get("name")), env.get("data"))
            for env in envelopes
        }
        collected_names = {str(env.get("name") or "") for env in envelopes}
        has_redis_tools = any(
            operation in operation_to_name
            for operation in {
                "info",
                "client_list",
                "slowlog",
                "memory_stats",
                "config_get",
                "replication_info",
            }
        )

        if not has_redis_tools:
            resolve_name = operation_to_name.get("resolve_redis_targets")
            if resolve_name and "resolve_redis_targets" not in collected_names:
                return AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call(
                            resolve_name,
                            {
                                "query": target_hint or query,
                                "allow_multiple": False,
                                "max_results": 5,
                                "attach_tools": True,
                                "preferred_capabilities": ["diagnostics"],
                            },
                        )
                    ],
                )
            return AIMessage(content=_fake_final_text(query, self.agent_kind))

        pending: List[Dict[str, Any]] = []
        for operation, args in _diagnostic_goals(self.agent_kind):
            tool_name = operation_to_name.get(operation)
            if not tool_name:
                continue
            signature = _operation_signature(tool_name, args)
            if signature in collected:
                continue
            pending.append(_tool_call(tool_name, args))
            if len(pending) >= self.max_tool_calls_per_turn:
                break

        if pending:
            return AIMessage(content="", tool_calls=pending)

        if self.agent_kind == "triage" and "search_index_info" not in collected_names:
            index_name = self._first_search_index(envelopes)
            tool_name = operation_to_name.get("search_index_info")
            if tool_name and index_name:
                return AIMessage(
                    content="",
                    tool_calls=[_tool_call(tool_name, {"index_name": index_name})],
                )

        return AIMessage(content=_fake_final_text(query, self.agent_kind))

    @staticmethod
    def _first_search_index(envelopes: Sequence[Dict[str, Any]]) -> Optional[str]:
        for envelope in reversed(list(envelopes or [])):
            if envelope.get("name") != "search_indexes":
                continue
            data = envelope.get("data")
            if not isinstance(data, dict):
                continue
            indexes = data.get("indexes")
            if isinstance(indexes, list) and indexes:
                return str(indexes[0])
        return None
