"""SRE LangGraph Agent 主链。

original 项目的 SRELangGraphAgent 是深度诊断入口：每次查询按上下文创建
ToolManager，把工具绑定给 LLM，经过 StateGraph 的 agent/tool loop 收集
evidence，再生成最终诊断回答。裁剪版保留这条主链；真实 LLM、checkpoint、
safety corrector 等平台能力只保留插槽。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, NotRequired, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from redis_sre_agent.core.clusters import RedisCluster
from redis_sre_agent.core.config import settings
from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.core.targets import TargetBinding, get_target_bindings_from_context
from redis_sre_agent.tools.manager import ToolManager

from .helpers import (
    NullEmitter,
    build_adapters_for_tooldefs,
    build_graph_config,
    build_result_envelope,
    extract_last_ai_response,
    guarded_ainvoke,
    resolve_graph_thread_id,
)
from .models import AgentResponse
from .terminal_synthesis import build_deterministic_diagnostic_response
from .tool_execution import execute_tool_calls_with_gate

logger = logging.getLogger(__name__)


SRE_SYSTEM_PROMPT = """You are a Redis SRE deep triage agent.

Use the available Redis tools to gather live diagnostic evidence before drawing
conclusions. Prefer read-only diagnostics, keep tool calls iterative, and make
the final answer evidence-backed.
"""


class AgentState(TypedDict):
    """SRE workflow 的状态 schema，主要字段沿用 original。"""

    messages: List[BaseMessage]
    session_id: str
    user_id: Optional[str]
    current_tool_calls: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int
    startup_system_prompt: Optional[str]
    startup_prompt_initialized: NotRequired[bool]
    instance_context: Optional[Dict[str, Any]]
    toolset_generation: NotRequired[int]
    signals_envelopes: List[Dict[str, Any]]


class SRELangGraphAgent:
    """基于 StateGraph 的 Redis 深度诊断 Agent。"""

    def __init__(
        self,
        redis_instance: Optional[RedisInstance] = None,
        redis_cluster: Optional[RedisCluster] = None,
        progress_emitter: Optional[Any] = None,
        exclude_mcp_categories: Optional[List[Any]] = None,
        support_package_path: Optional[Path] = None,
        llm: Optional[Any] = None,
        **_: Any,
    ):
        self.settings = settings
        self.redis_instance = redis_instance
        self.redis_cluster = redis_cluster
        self.exclude_mcp_categories = exclude_mcp_categories
        self.support_package_path = support_package_path
        self._progress_emitter = (
            progress_emitter if progress_emitter is not None else NullEmitter()
        )
        if llm is None:
            from ._compat import FakeToolCallingLLM

            llm = FakeToolCallingLLM(agent_kind="triage")
        self.llm = llm
        self.llm_with_tools = self.llm

    def _build_workflow(
        self,
        tool_mgr: ToolManager,
        target_instance: Optional[RedisInstance] = None,
    ) -> StateGraph:
        """构建 SRE triage workflow。"""

        runtime_tools_by_generation: Dict[tuple[int, int], Dict[str, Any]] = {}

        async def ensure_runtime_tools(
            requested_generation: Optional[int] = None,
        ) -> Dict[str, Any]:
            current_generation = tool_mgr.get_toolset_generation()
            generation = requested_generation or current_generation
            if generation != current_generation:
                generation = current_generation
            try:
                tooldefs = tool_mgr.get_tools_for_llm(max_tools=64)
            except TypeError:
                tooldefs = tool_mgr.get_tools_for_llm()
            except AttributeError:
                tooldefs = tool_mgr.get_tools()
            cache_key = (generation, len(tooldefs))
            cached = runtime_tools_by_generation.get(cache_key)
            if cached is not None:
                return cached
            adapters = await build_adapters_for_tooldefs(tool_mgr, tooldefs)
            llm_tools = adapters or tooldefs
            llm_with_tools = (
                self.llm.bind_tools(llm_tools) if hasattr(self.llm, "bind_tools") else self.llm
            )
            runtime = {
                "generation": generation,
                "tooldefs_by_name": {tool.name: tool for tool in tooldefs},
                "llm_with_tools": llm_with_tools,
            }
            runtime_tools_by_generation[cache_key] = runtime
            return runtime

        async def agent_node(state: AgentState) -> Dict[str, Any]:
            """调用 LLM 决定下一批工具或最终回答。"""

            runtime = await ensure_runtime_tools()
            messages = list(state.get("messages") or [])
            iteration_count = state.get("iteration_count", 0)
            startup_system_prompt = state.get("startup_system_prompt") or SRE_SYSTEM_PROMPT
            startup_prompt_initialized = state.get("startup_prompt_initialized", False)

            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=startup_system_prompt)] + messages
                startup_prompt_initialized = True

            response = await guarded_ainvoke(
                runtime["llm_with_tools"],
                messages,
                request_kind="langgraph_agent.agent_node",
            )
            if not isinstance(response, AIMessage):
                response = AIMessage(
                    content=getattr(response, "content", response),
                    tool_calls=list(getattr(response, "tool_calls", []) or []),
                )

            return {
                "messages": list(state.get("messages") or []) + [response],
                "iteration_count": iteration_count + 1,
                "startup_system_prompt": startup_system_prompt,
                "startup_prompt_initialized": startup_prompt_initialized,
                "toolset_generation": runtime["generation"],
                "current_tool_calls": list(response.tool_calls or []),
                "instance_context": state.get("instance_context"),
                "signals_envelopes": list(state.get("signals_envelopes") or []),
            }

        async def tool_node(state: AgentState) -> Dict[str, Any]:
            """执行工具并把 ToolMessage / ResultEnvelope 写回状态。"""

            runtime = await ensure_runtime_tools(state.get("toolset_generation"))
            tooldefs_by_name = runtime["tooldefs_by_name"]
            messages = list(state.get("messages") or [])
            tool_calls = list(state.get("current_tool_calls") or [])
            if not tool_calls and messages and isinstance(messages[-1], AIMessage):
                tool_calls = list(messages[-1].tool_calls or [])
            if not tool_calls:
                return {}

            for call in tool_calls:
                tool_name = str(call.get("name") or "")
                tool_args = call.get("args") if isinstance(call.get("args"), dict) else {}
                status_msg = None
                if tool_name and hasattr(tool_mgr, "get_status_update"):
                    status_msg = tool_mgr.get_status_update(tool_name, tool_args)
                if status_msg and hasattr(self._progress_emitter, "emit"):
                    await self._progress_emitter.emit(status_msg, "agent_reflection")

            tool_messages = await execute_tool_calls_with_gate(
                tool_manager=tool_mgr,
                tool_calls=tool_calls,
            )
            envelopes = list(state.get("signals_envelopes") or [])
            for index, call in enumerate(tool_calls):
                tool_name = str(call.get("name") or f"tool_{index + 1}")
                tool_args = call.get("args") if isinstance(call.get("args"), dict) else {}
                tool_message = tool_messages[index] if index < len(tool_messages) else None
                if isinstance(tool_message, ToolMessage):
                    envelopes.append(
                        build_result_envelope(
                            tool_name,
                            tool_args,
                            tool_message,
                            tooldefs_by_name,
                        )
                    )

            await self._attach_target_tools_from_resolution(tool_mgr, envelopes)

            return {
                "messages": messages + tool_messages,
                "current_tool_calls": [],
                "toolset_generation": tool_mgr.get_toolset_generation(),
                "signals_envelopes": envelopes,
            }

        async def reasoning_node(state: AgentState) -> Dict[str, Any]:
            """从已收集 evidence 生成最终回答。"""

            messages = list(state.get("messages") or [])
            envelopes = list(state.get("signals_envelopes") or [])
            query = ""
            for message in reversed(messages):
                if isinstance(message, HumanMessage):
                    query = str(message.content)
                    break
            response = AIMessage(
                content=build_deterministic_diagnostic_response(
                    query,
                    envelopes,
                    agent_kind="triage",
                )
            )
            return {"messages": messages + [response], "signals_envelopes": envelopes}

        def should_continue(state: AgentState) -> str:
            """决定继续工具循环、进入 reasoning，或结束。"""

            iteration_count = state.get("iteration_count", 0)
            max_iterations = state.get("max_iterations", self.settings.max_iterations)
            messages = state.get("messages") or []
            if iteration_count >= max_iterations:
                logger.warning("SRE agent reached max iterations (%s)", max_iterations)
                return "reasoning"
            if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
                return "tools"
            if state.get("current_tool_calls"):
                return "tools"
            if any(isinstance(message, ToolMessage) for message in messages):
                return "reasoning"
            return END

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("reasoning", reasoning_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "reasoning": "reasoning", END: END},
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("reasoning", END)
        return workflow

    async def _attach_target_tools_from_resolution(
        self,
        tool_mgr: ToolManager,
        envelopes: List[Dict[str, Any]],
    ) -> None:
        """target discovery fallback：确保解析出的 binding 已挂载到 ToolManager。"""

        if tool_mgr.get_tools_by_provider_names(["redis_command"]):
            return
        bindings: List[TargetBinding] = []
        generation: Optional[int] = None
        for envelope in envelopes:
            if envelope.get("name") != "resolve_redis_targets":
                continue
            data = envelope.get("data")
            if not isinstance(data, dict):
                continue
            if data.get("toolset_generation") is not None:
                try:
                    generation = int(data.get("toolset_generation"))
                except Exception:
                    generation = generation
            for raw_binding in data.get("target_bindings") or []:
                try:
                    bindings.append(TargetBinding.model_validate(raw_binding))
                except Exception:
                    continue
        if bindings:
            await tool_mgr.attach_bound_targets(bindings, generation=generation)

    def _enhance_query(self, query: str, context: Dict[str, Any]) -> str:
        if self.redis_instance is not None:
            return f"""User Query: {query}

IMPORTANT CONTEXT: This query is specifically about Redis instance:
- Instance ID: {self.redis_instance.id}
- Instance Name: {self.redis_instance.name}
- Environment: {self.redis_instance.environment}
- Usage: {self.redis_instance.usage}
- Instance Type: {self.redis_instance.instance_type}

Your diagnostic tools are pre-configured for this instance."""
        if self.redis_cluster is not None:
            return f"""User Query: {query}

IMPORTANT CONTEXT: This query is scoped to Redis cluster:
- Cluster ID: {self.redis_cluster.id}
- Cluster Name: {self.redis_cluster.name}
- Environment: {self.redis_cluster.environment}
- Cluster Type: {self.redis_cluster.cluster_type}"""
        target_query = str(context.get("target_query") or context.get("target") or "").strip()
        if target_query and target_query != query:
            return f"Target Hint: {target_query}\n\nUser Query: {query}"
        return query

    async def _process_query(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str],
        max_iterations: int,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[BaseMessage]] = None,
        progress_emitter: Optional[Any] = None,
    ) -> AgentResponse:
        normalized_context = dict(context or {})
        if progress_emitter is not None:
            self._progress_emitter = progress_emitter
        thread_id = str(normalized_context.get("thread_id") or session_id or "triage")
        initial_bindings = get_target_bindings_from_context(normalized_context)
        support_package_path = self.support_package_path or normalized_context.get(
            "support_package_path"
        )
        if isinstance(support_package_path, str):
            support_package_path = Path(support_package_path)

        async with ToolManager(
            redis_instance=self.redis_instance,
            redis_cluster=self.redis_cluster,
            initial_target_bindings=initial_bindings or None,
            exclude_mcp_categories=self.exclude_mcp_categories,
            support_package_path=support_package_path,
            thread_id=thread_id,
            task_id=normalized_context.get("task_id"),
            user_id=user_id,
            graph_type="redis_triage",
        ) as tool_mgr:
            self.workflow = self._build_workflow(tool_mgr, self.redis_instance)
            enhanced_query = self._enhance_query(query, normalized_context)
            initial_messages: List[BaseMessage] = [SystemMessage(content=SRE_SYSTEM_PROMPT)]
            if conversation_history:
                initial_messages.extend(conversation_history)
            initial_messages.append(HumanMessage(content=enhanced_query))
            initial_state: AgentState = {
                "messages": initial_messages,
                "session_id": session_id,
                "user_id": user_id,
                "current_tool_calls": [],
                "iteration_count": 0,
                "max_iterations": max_iterations,
                "startup_system_prompt": SRE_SYSTEM_PROMPT,
                "startup_prompt_initialized": True,
                "instance_context": normalized_context or None,
                "toolset_generation": tool_mgr.get_toolset_generation(),
                "signals_envelopes": [],
            }
            graph_thread_id = resolve_graph_thread_id(session_id, normalized_context)
            app = self.workflow.compile()
            self.app = app
            final_state = await app.ainvoke(
                initial_state,
                config=build_graph_config(
                    graph_thread_id=graph_thread_id,
                    recursion_limit=getattr(self.settings, "recursion_limit", 100),
                ),
            )
            tool_envelopes = list(final_state.get("signals_envelopes") or [])
            messages = list(final_state.get("messages") or [])
            response_text = extract_last_ai_response(messages, terminal_only=True)
            if not response_text:
                response_text = build_deterministic_diagnostic_response(
                    query,
                    tool_envelopes,
                    agent_kind="triage",
                )
            return AgentResponse(response=response_text, tool_envelopes=tool_envelopes)

    async def process_query(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str],
        max_iterations: int = settings.max_iterations,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[BaseMessage]] = None,
        progress_emitter: Optional[Any] = None,
    ) -> AgentResponse:
        """处理一次 SRE 查询。"""

        logger.info("Processing SRE query for user %s", user_id or "<anonymous>")
        try:
            return await self._process_query(
                query=query,
                session_id=session_id,
                user_id=user_id,
                max_iterations=max_iterations,
                context=context,
                conversation_history=conversation_history,
                progress_emitter=progress_emitter,
            )
        except Exception as exc:
            logger.exception("SRE agent error: %s", exc)
            return AgentResponse(response=f"Error processing query: {exc}")

    async def resume_query(
        self,
        *,
        session_id: str,
        user_id: Optional[str],
        context: Optional[Dict[str, Any]] = None,
        progress_emitter: Optional[Any] = None,
        resume_payload: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """checkpoint/resume 是后续阶段插槽。"""

        return AgentResponse(
            response="resume is a future-stage slot; 当前阶段只恢复同步 Agent 诊断主链。",
            tool_envelopes=[],
        )


def get_sre_agent(*args: Any, **kwargs: Any) -> SRELangGraphAgent:
    """每次创建新的 triage agent，避免跨任务状态串扰。"""

    return SRELangGraphAgent(*args, **kwargs)
