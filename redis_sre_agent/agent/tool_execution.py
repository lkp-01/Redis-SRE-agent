"""Agent 工具执行封装。

original 项目里 tool node 不直接拼报告，而是把 LLM 请求的 tool_calls 交给
ToolManager，再把结果作为 ToolMessage 放回图状态。这里保留同样的生命周期，
审批、人审恢复等更复杂 gate 仍是后续插槽。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import ToolMessage

_REDACTED = "[REDACTED]"


def _safe_error_message(exc: Exception) -> str:
    message = str(exc)
    message = re.sub(
        r"(?i)\b(rediss?|unix)://[^\s'\"<>@]+@[^\s'\"<>]+",
        lambda match: re.sub(r"://([^:@/]+):[^@/]+@", r"://\1:[REDACTED]@", match.group(0)),
        message,
    )
    message = re.sub(
        r"(?i)\b(password|secret|token|requirepass|masterauth|pass)(\s*[=:]\s*)([^\s,;}]+)",
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}",
        message,
    )
    return message


async def execute_tool_call_with_gate(
    *,
    tool_manager: Any,
    tool_name: str,
    tool_args: Dict[str, Any],
    local_tools: Optional[Dict[str, Any]] = None,
) -> Any:
    """执行单个工具调用，保留后续审批 gate 的接缝。"""

    local_tool = (local_tools or {}).get(tool_name)
    if local_tool is not None:
        result = local_tool(**dict(tool_args or {}))
        if hasattr(result, "__await__"):
            return await result
        return result
    return await tool_manager.resolve_tool_call(tool_name, dict(tool_args or {}))


def _tool_call_id(tool_call: Dict[str, Any]) -> str:
    return str(
        tool_call.get("id")
        or tool_call.get("tool_call_id")
        or tool_call.get("call_id")
        or ""
    )


def _tool_call_name(tool_call: Dict[str, Any]) -> str:
    name = tool_call.get("name")
    if not name and isinstance(tool_call.get("function"), dict):
        name = tool_call["function"].get("name")
    return str(name or "")


def _tool_call_args(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    args = tool_call.get("args")
    if args is None and isinstance(tool_call.get("function"), dict):
        raw_arguments = tool_call["function"].get("arguments")
        if isinstance(raw_arguments, str):
            try:
                args = json.loads(raw_arguments or "{}")
            except Exception:
                args = {}
        elif isinstance(raw_arguments, dict):
            args = raw_arguments
    return dict(args or {}) if isinstance(args, dict) else {}


def _message_content(result: Any) -> str:
    try:
        return json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return json.dumps({"status": "success", "raw": str(result)}, ensure_ascii=False)


async def execute_tool_calls_with_gate(
    *,
    tool_manager: Any,
    tool_calls: List[Dict[str, Any]],
    local_tools: Optional[Dict[str, Any]] = None,
) -> List[ToolMessage]:
    """批量执行工具调用，并返回可追加到 LangGraph 状态的 ToolMessage。"""

    messages: List[ToolMessage] = []
    for index, tool_call in enumerate(tool_calls or []):
        name = _tool_call_name(tool_call)
        args = _tool_call_args(tool_call)
        call_id = _tool_call_id(tool_call) or f"tool_call_{index + 1}"
        if not name:
            payload = {"status": "failed", "error": "missing tool name"}
            messages.append(ToolMessage(_message_content(payload), tool_call_id=call_id))
            continue
        try:
            result = await execute_tool_call_with_gate(
                tool_manager=tool_manager,
                tool_name=name,
                tool_args=args,
                local_tools=local_tools,
            )
            messages.append(
                ToolMessage(_message_content(result), tool_call_id=call_id, name=name)
            )
        except Exception as exc:
            payload = {"status": "failed", "error": _safe_error_message(exc)}
            messages.append(ToolMessage(_message_content(payload), tool_call_id=call_id, name=name))
    return messages
