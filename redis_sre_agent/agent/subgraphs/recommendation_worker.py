"""按主题生成结构化 Recommendation 的裁剪版 original 子图。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from ..helpers import build_result_envelope, guarded_ainvoke
from ..models import Recommendation


class RecState(TypedDict, total=False):
    messages: List[BaseMessage]
    budget: int
    topic: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    instance: Dict[str, Any]
    result: Dict[str, Any]
    knowledge_envelopes: List[Dict[str, Any]]


def build_recommendation_worker(
    base_llm: Any,
    knowledge_tool_adapters: List[Any],
    *,
    knowledge_tooldefs_by_name: Optional[Dict[str, Any]] = None,
    max_tool_steps: int = 3,
) -> Any:
    """保留 original 的短工具循环，再用 structured output 生成 Recommendation。"""

    adapters = list(knowledge_tool_adapters or [])
    knowledge_definitions = dict(knowledge_tooldefs_by_name or {})
    tool_node = ToolNode(adapters)
    llm_with_tools = base_llm.bind_tools(adapters) if adapters else base_llm

    async def llm_node(state: RecState) -> RecState:
        messages = list(state.get("messages") or [])
        response = await guarded_ainvoke(
            llm_with_tools,
            messages,
            request_kind="recommendation_worker.loop",
        )
        if not isinstance(response, AIMessage):
            response = AIMessage(
                content=getattr(response, "content", response),
                tool_calls=list(getattr(response, "tool_calls", []) or []),
            )
        return {
            **state,
            "messages": messages + [response],
            "budget": int(state.get("budget", max_tool_steps)),
        }

    async def tools_node(state: RecState) -> RecState:
        previous = list(state.get("messages") or [])
        last_ai: Optional[AIMessage] = next(
            (message for message in reversed(previous) if isinstance(message, AIMessage)),
            None,
        )
        tool_calls = list(last_ai.tool_calls or []) if last_ai else []
        output = await tool_node.ainvoke({"messages": previous})
        tool_messages = [
            message for message in output.get("messages", []) if isinstance(message, ToolMessage)
        ]
        calls_by_id = {
            str(call.get("id") or call.get("tool_call_id") or ""): call
            for call in tool_calls
        }
        knowledge_envelopes = list(state.get("knowledge_envelopes") or [])
        for index, tool_message in enumerate(tool_messages):
            call_id = str(getattr(tool_message, "tool_call_id", "") or "")
            call = calls_by_id.get(call_id)
            if call is None and index < len(tool_calls):
                call = tool_calls[index]
            call = call or {}
            tool_name = str(
                call.get("name") or getattr(tool_message, "name", "") or ""
            )
            if tool_name not in knowledge_definitions:
                continue
            tool_args = call.get("args") if isinstance(call.get("args"), dict) else {}
            knowledge_envelopes.append(
                build_result_envelope(
                    tool_name,
                    tool_args,
                    tool_message,
                    knowledge_definitions,
                )
            )
        return {
            **state,
            "messages": previous + tool_messages,
            "budget": max(0, int(state.get("budget", max_tool_steps)) - 1),
            "knowledge_envelopes": knowledge_envelopes,
        }

    def should_continue(state: RecState) -> str:
        last_ai: Optional[AIMessage] = next(
            (
                message
                for message in reversed(state.get("messages", []))
                if isinstance(message, AIMessage)
            ),
            None,
        )
        return (
            "tools"
            if last_ai and last_ai.tool_calls and int(state.get("budget", max_tool_steps)) > 0
            else "synth"
        )

    async def synth_node(state: RecState) -> RecState:
        structured_llm = base_llm.with_structured_output(
            Recommendation,
            method="function_calling",
        )
        topic = state.get("topic") or {}
        evidence = state.get("evidence") or []
        knowledge_evidence = state.get("knowledge_envelopes") or []
        instance = state.get("instance") or {}
        system = SystemMessage(
            content=(
                "You are producing operator-facing recommendations. Use only the supplied "
                "evidence and any local tool results. Do not invent facts, commands, APIs, or "
                "sources. If evidence is insufficient, add an investigation step. Output must "
                "match the Recommendation schema."
            )
        )
        human = HumanMessage(
            content=(
                f"Topic (JSON):\n{topic}\n\n"
                f"Instance Facts (JSON):\n{instance}\n\n"
                "Evidence is a verbatim or summarized record of upstream tool calls.\n"
                f"Evidence (JSON):\n{evidence}\n\n"
                f"Knowledge Evidence (ResultEnvelope JSON):\n{knowledge_evidence}"
            )
        )
        recommendation = await guarded_ainvoke(
            structured_llm,
            [system, human],
            request_kind="recommendation_worker.synth",
        )
        result = (
            dict(recommendation)
            if isinstance(recommendation, dict)
            else recommendation.model_dump()
        )
        result["topic_id"] = topic.get("id", "T?")
        return {**state, "result": result}

    graph = StateGraph(RecState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tools_node)
    graph.add_node("synth", synth_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", "synth": "synth"})
    graph.add_edge("tools", "llm")
    graph.add_edge("synth", END)
    return graph.compile()
