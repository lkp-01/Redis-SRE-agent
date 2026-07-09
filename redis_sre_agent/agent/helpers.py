"""Agent 层的轻量 helper。

原项目在这里集中处理工具结果 envelope、回答文本归一化和 citation 提取。Stage 5
只保留诊断主链路需要的部分，避免引入 LangChain、JMESPath 等后续阶段依赖。
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from .models import ResultEnvelope

KNOWLEDGE_SEARCH_RETRIEVAL_KIND = "knowledge_search"
KNOWLEDGE_SEARCH_RETRIEVAL_LABEL = "Knowledge search"

_SUMMARY_THRESHOLD = 1200
_KNOWN_TOOL_OPERATIONS = (
    "list_known_redis_targets",
    "resolve_redis_targets",
    "search_index_info",
    "replication_info",
    "cluster_info",
    "memory_stats",
    "client_list",
    "config_get",
    "sample_keys",
    "search_indexes",
    "acl_log",
    "slowlog",
    "info",
    "search",
)


class NullEmitter:
    """进度事件的 no-op 实现，用在 CLI/test 没有传 emitter 的场景。"""

    async def emit(
        self,
        message: str,
        event_type: str = "info",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        return None


def build_graph_config(
    *,
    graph_thread_id: str,
    recursion_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """构造 LangGraph ainvoke config，字段名贴近 original checkpoint 调用。"""

    config: Dict[str, Any] = {"configurable": {"thread_id": graph_thread_id}}
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    return config


def resolve_graph_thread_id(session_id: str, context: Optional[Dict[str, Any]] = None) -> str:
    """从 query context 中解析 LangGraph 使用的 thread_id。"""

    context = context or {}
    return str(context.get("thread_id") or session_id or "agent-thread")


def coerce_response_text(content: Any) -> str:
    """把模型或 helper 返回的多种内容形态统一成非空字符串。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def extract_last_ai_response(messages: List[Any], *, terminal_only: bool = False) -> str:
    """取最后一条非空 AI 回复，行为贴近 original helper。"""

    if not messages:
        return ""
    if not terminal_only:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                text = coerce_response_text(getattr(message, "content", ""))
                if text:
                    return text
        return ""

    saw_trailing_ai = False
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            if saw_trailing_ai:
                break
            return ""
        saw_trailing_ai = True
        text = coerce_response_text(getattr(message, "content", ""))
        if text:
            return text
    return ""


async def guarded_ainvoke(llm: Any, messages: List[Any], **_: Any) -> Any:
    """裁剪版 LLM 调用入口。

    original 这里会接入 request guard、重试和 token 保护；当前阶段只保留统一调用点，
    测试可以传 fake LLM，正式链路仍走 LangChain message/runtime。
    """

    if hasattr(llm, "ainvoke"):
        return await llm.ainvoke(messages)
    if callable(llm):
        result = llm(messages)
        if inspect.isawaitable(result):
            return await result
        return result
    raise TypeError("LLM object must provide ainvoke() or be callable.")


async def build_adapters_for_tooldefs(tool_manager: Any, tooldefs: List[Any]) -> list[Any]:
    """把 ToolDefinition 转成 LangChain StructuredTool adapter。

    Agent 把这些 adapter 绑定给 LLM；真正执行时仍回到 ToolManager，不绕过 provider。
    """

    try:
        from typing import Any as _Any

        from langchain_core.tools import StructuredTool as _StructuredTool
        from pydantic import BaseModel as _BaseModel
        from pydantic import ConfigDict as _ConfigDict
        from pydantic import Field as _Field
        from pydantic import create_model as _create_model
    except Exception:
        return []

    def _field_default(spec: dict, is_required: bool):
        if is_required:
            return ...
        if "default" in (spec or {}):
            return spec.get("default")
        return None

    json_type_map: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def _python_type_for(spec: dict, is_required: bool):
        raw_type = (spec or {}).get("type")
        if isinstance(raw_type, list):
            concrete = next((t for t in raw_type if isinstance(t, str) and t != "null"), None)
            py_type = json_type_map.get(concrete, _Any)
            nullable = (not is_required) or ("null" in raw_type)
        elif isinstance(raw_type, str):
            py_type = json_type_map.get(raw_type, _Any)
            nullable = not is_required
        else:
            py_type = _Any
            nullable = not is_required
        if nullable and py_type is not _Any:
            py_type = Optional[py_type]
        return py_type

    def _args_model_from_parameters(tool_name: str, params: dict) -> type[_BaseModel]:
        props = (params or {}).get("properties", {}) or {}
        required = set((params or {}).get("required", []) or [])
        fields: dict[str, tuple[_Any, _Any]] = {}
        for key, spec in props.items():
            spec = spec or {}
            default = _field_default(spec, key in required)
            fields[key] = (
                _python_type_for(spec, key in required),
                _Field(default, description=spec.get("description")),
            )
        args_model = _create_model(f"{tool_name}_Args", __base__=_BaseModel, **fields)
        try:
            args_model.model_config = _ConfigDict(extra="allow")  # type: ignore[attr-defined]
        except Exception:
            pass
        return args_model

    adapters: list[Any] = []
    for tool_def in tooldefs or []:

        async def _exec_fn(_name=tool_def.name, **kwargs):
            from .tool_execution import execute_tool_call_with_gate

            return await execute_tool_call_with_gate(
                tool_manager=tool_manager,
                tool_name=_name,
                tool_args=kwargs or {},
            )

        args_model = _args_model_from_parameters(tool_def.name, tool_def.parameters or {})
        adapters.append(
            _StructuredTool.from_function(
                coroutine=_exec_fn,
                name=tool_def.name,
                description=tool_def.description or "",
                args_schema=args_model,
            )
        )
    return adapters


def extract_tool_operation_name(tool_name: str) -> str:
    """从动态工具名中取出稳定操作名。

    ToolProvider 会生成类似 `redis_command_abc123_memory_stats` 的名字；Agent 报告里
    只需要 `memory_stats` 这类操作名。
    """
    if not tool_name:
        return "tool"
    short_name = tool_name.rsplit(".", 1)[-1]
    match = re.search(r"_[0-9a-f]{6}_(.+)$", short_name)
    if match:
        return match.group(1)
    for operation in _KNOWN_TOOL_OPERATIONS:
        if short_name == operation or short_name.endswith(f"_{operation}"):
            return operation
    return short_name


def _description_for_tool(tool_def: Any) -> Optional[str]:
    if tool_def is None:
        return None
    if isinstance(tool_def, dict):
        description = tool_def.get("description")
    else:
        description = getattr(tool_def, "description", None)
    return str(description) if description is not None else None


def _parse_tool_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    content = getattr(result, "content", result)
    if isinstance(content, dict):
        return dict(content)
    if isinstance(content, str):
        stripped = content.strip()
        if stripped:
            try:
                first_brace = stripped.find("{")
                payload = stripped[first_brace:] if first_brace >= 0 else stripped
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    return parsed
                return {"raw": parsed}
            except Exception:
                return {"raw": stripped[:4000]}
    return {"raw": str(content)[:4000]}


def _summary_for_data(data: Dict[str, Any]) -> Optional[str]:
    try:
        payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        payload = str(data)
    if len(payload) <= _SUMMARY_THRESHOLD:
        return None
    return f"{payload[:_SUMMARY_THRESHOLD].rstrip()}... [truncated {len(payload) - _SUMMARY_THRESHOLD} chars]"


def build_result_envelope(
    tool_name: str,
    args: Dict[str, Any],
    result: Any,
    tooldefs_by_name: Dict[str, Any],
) -> Dict[str, Any]:
    """把一次工具调用结果打包成原项目 `ResultEnvelope` 形状。"""
    data_obj = _parse_tool_result(result)
    raw_status = str(data_obj.get("status", "")).lower()
    envelope_status = "error" if raw_status in {"error", "failed", "failure"} else "success"
    tool_def = tooldefs_by_name.get(tool_name) if tool_name else None
    envelope = ResultEnvelope(
        tool_key=tool_name or "tool",
        name=extract_tool_operation_name(tool_name or "tool"),
        description=_description_for_tool(tool_def),
        args=dict(args or {}),
        status=envelope_status,
        data=data_obj,
        summary=_summary_for_data(data_obj),
    )
    return envelope.model_dump(mode="json")


def extract_citations(envelopes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从 knowledge 工具 envelope 中提取 citation。

    Stage 5 的 knowledge provider 只是 dummy，占位结果通常为空；这里保留原项目的派生
    机制，后续真实 RAG 接回时不需要改 `AgentResponse`。
    """
    citations: List[Dict[str, Any]] = []
    for envelope in envelopes or []:
        tool_key = str(envelope.get("tool_key", ""))
        name = str(envelope.get("name", ""))
        if "knowledge" not in tool_key.lower():
            continue
        data = envelope.get("data") or {}
        if not isinstance(data, dict):
            continue
        results = data.get("results") or []
        if not isinstance(results, list):
            continue
        default_retrieval_kind = str(data.get("retrieval_kind") or "").strip()
        default_retrieval_label = str(data.get("retrieval_label") or "").strip()
        if "search" in tool_key.lower() or "search" in name.lower():
            default_retrieval_kind = default_retrieval_kind or KNOWLEDGE_SEARCH_RETRIEVAL_KIND
            default_retrieval_label = default_retrieval_label or KNOWLEDGE_SEARCH_RETRIEVAL_LABEL
        for result in results:
            if not isinstance(result, dict):
                continue
            citation = dict(result)
            if default_retrieval_kind:
                citation.setdefault("retrieval_kind", default_retrieval_kind)
            if default_retrieval_label:
                citation.setdefault("retrieval_label", default_retrieval_label)
            citations.append(citation)
    return citations
