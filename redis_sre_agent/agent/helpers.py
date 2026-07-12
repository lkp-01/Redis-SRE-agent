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

# 组装 app.ainvoke() 需要的 config
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

# 从 query context 中解析 LangGraph 使用的 thread_id。
def resolve_graph_thread_id(session_id: str, context: Optional[Dict[str, Any]] = None) -> str:

    context = context or {}
    return str(context.get("thread_id") or session_id or "agent-thread")

# 把文本转成JSON
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

# 揪出最后一条非空的 AI 回复文本。
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

# 暂时加上的，可以用来防假LLM
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

#把自研的标准工具定义（ToolDefinition）动态翻译成 LangChain 认识的 StructuredTool 适配器
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
    """
    【大白话功能】数据解包员：把工具返回的各种奇形怪状的原始数据，强行揉碎、拼装成标准的 Python 字典。
    """
    # 1. 如果原始结果本身就已经是一个标准字典了，直接复制一份返回
    if isinstance(result, dict):
        return dict(result)

    # 2. 如果不是字典，看看它是不是一个对象（比如 LangChain 的 Message 对象）
    # 尝试读取它的 .content 属性；如果没有这个属性，就依旧用 result 自身
    content = getattr(result, "content", result)

    # 3. 检查拿到的 content 是不是字典，如果是，同样复制一份返回
    if isinstance(content, dict):
        return dict(content)

    # 4. 如果 content 是一串文本（比如大模型或者某个工具吐出的 JSON 字符串）
    if isinstance(content, str):
        # 先扒掉两边的空格和换行
        stripped = content.strip()

        # 如果剥离后文本不为空，尝试把它当做 JSON 字符串来解密
        if stripped:
            try:
                # 骚操作：寻找文本中第一个左大括号 "{" 的位置，用来兼容那些开头带了废话的文本
                # （比如工具返回了："这是结果：{"status": "ok"}"，能精准定位到 "{"）
                first_brace = stripped.find("{")

                # 如果找到了 "{"，就切片从 "{" 开始往后的所有内容；没找到就用完整的文本
                payload = stripped[first_brace:] if first_brace >= 0 else stripped

                # 尝试把这段文本反序列化成 Python 对象
                parsed = json.loads(payload)

                # 如果解密出来确实是一个标准的字典，开心地直接返回它
                if isinstance(parsed, dict):
                    return parsed

                # 如果解密出来是列表或者其他东西（不是字典），就用 {"raw": 结果} 打包返回
                return {"raw": parsed}
            except Exception:
                # 如果上面 JSON 解密失败（说明只是一串普通文本），就截取前 4000 个字符，打包成字典返回
                return {"raw": stripped[:4000]}

    # 5. 兜底防御：如果既不是字典也不是字符串，强行把它转成纯文本，截取前 4000 字符打包返回
    return {"raw": str(content)[:4000]}


def _summary_for_data(data: Dict[str, Any]) -> Optional[str]:
    """
    【大白话功能】数据裁剪员：检查数据是不是太胖（太长）了，如果太胖就无情截断，并打上“已裁剪”的标签。
    """
    try:
        # 1. 尝试把刚才解包好的字典重新转成一串紧凑的 JSON 字符串
        # ensure_ascii=False 保证中文不乱码；sort_keys=True 让 key 排序方便人类阅读
        payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        # 2. 如果转 JSON 失败（比如里面有无法序列化的古怪对象），就用 str() 强行转成普通文本
        payload = str(data)

    # 3. 检查这串文本的长度有没有超过设定的警戒线（代码前文定义的 _SUMMARY_THRESHOLD = 1200 字符）
    if len(payload) <= _SUMMARY_THRESHOLD:
        # 如果很短很安全，说明不需要裁剪，直接返回 None（大管家看到 None 就知道不需要做摘要）
        return None

    # 4. 如果不幸超长了，无情地举起剪刀：
    # 截取前 1200 个字符，剥离右侧空格，然后拼上标志性的尾巴："... [truncated 剩余字符数 chars]"
    # 这样大模型看到尾巴就知道：“哦，数据太长被后勤部给切了，我拿到的只是前面的精简版。”
    return f"{payload[:_SUMMARY_THRESHOLD].rstrip()}... [truncated {len(payload) - _SUMMARY_THRESHOLD} chars]"


def build_result_envelope(
    tool_name: str,                       # 输入参数：被调用的工具完整名称（如：redis_command_abc123_info）
    args: Dict[str, Any],                 # 输入参数：调用工具时传入的参数字典
    result: Any,                          # 输入参数：工具执行完返回的原生结果数据（可能是对象、字符串或字典）
    tooldefs_by_name: Dict[str, Any],     # 输入参数：全局工具定义名册字典，用于查工具的描述信息
) -> Dict[str, Any]:                      # 返回值：打包好的、符合 ResultEnvelope 规范的标准 JSON 字典
    """把一次工具调用结果打包成原项目 `ResultEnvelope` 形状。"""

    # 1. 解析原生结果：把各种奇形怪状的 result（如 LangChain 消息对象、JSON 字符串等）统一清洗并解析成标准的 Python 字典
    data_obj = _parse_tool_result(result)

    # 2. 提取原始状态：从解析后的字典中尝试获取 'status' 字段，转成字符串并归一化为全小写
    raw_status = str(data_obj.get("status", "")).lower()

    # 3. 判定最终状态：如果原始状态里包含 "error", "failed", "failure" 中的任意一个，信封状态就定为 "error"，否则全部视为 "success"
    envelope_status = "error" if raw_status in {"error", "failed", "failure"} else "success"

    # 4. 获取工具定义：如果传入了工具名，就去工具名册字典里把该工具的静态定义（ToolDefinition）捞出来
    tool_def = tooldefs_by_name.get(tool_name) if tool_name else None

    # 5. 组装标准信封：实例化一个 Pydantic 的 ResultEnvelope 模型对象，把清洗好的数据一个个塞进去
    envelope = ResultEnvelope(
        tool_key=tool_name or "tool",                             # 注入工具的完整防伪键名，没传则兜底为 "tool"
        name=extract_tool_operation_name(tool_name or "tool"),    # 剥离工具名里的哈希前缀，只留下干净的操作名（如 info）
        description=_description_for_tool(tool_def),              # 从静态定义中提取出人类可读的工具功能描述
        args=dict(args or {}),                                    # 复制一份入参字典，防止外部修改
        status=envelope_status,                                   # 注入上面判定好的信封状态（success 或 error）
        data=data_obj,                                            # 注入解析好的完整原生数据字典
        summary=_summary_for_data(data_obj),                      # 数据体量太大时自动进行截断，生成一份缩略摘要供 LLM 阅读
    )

    # 6. 导出标准字典：把组装好的 Pydantic 模型对象转换（dump）成标准的 JSON 兼容字典并返回
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
