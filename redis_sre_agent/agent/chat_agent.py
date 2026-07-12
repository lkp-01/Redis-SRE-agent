"""轻量 ChatAgent。

这个文件按 original 的 chat agent 主链恢复：process_query 创建 ToolManager，
构建 StateGraph，agent node 产生 tool_calls，tool node 执行工具并把 evidence
放回状态，最后再由 agent node 生成回答。当前裁剪版正式 workflow 使用真实
LangGraph/LangChain message runtime；没有外部 LLM 时仅用 fake LLM 作为测试 fallback。
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
from .terminal_synthesis import (
    TerminalSynthesisConfig,
    build_deterministic_diagnostic_response,
    synthesize_terminal_response,
)
from .tool_execution import execute_tool_calls_with_gate

logger = logging.getLogger(__name__)


CHAT_SYSTEM_PROMPT = """You are a Redis SRE chat agent.

Work iteratively:
1. Decide which Redis diagnostic tools are needed.
2. Call a small number of tools.
3. Read the returned evidence before deciding whether to call more tools.
4. Answer only from collected evidence and clearly say when evidence is missing.
"""


class ChatAgentState(TypedDict):
    """Chat workflow 的状态形状，字段名沿用 original。"""

    messages: List[BaseMessage]
    session_id: str
    user_id: Optional[str]
    current_tool_calls: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int
    startup_system_prompt: Optional[str]
    startup_prompt_initialized: NotRequired[bool]
    toolset_generation: NotRequired[int]
    signals_envelopes: List[Dict[str, Any]]


class ChatAgent:
    """基于 StateGraph 的轻量 Redis 问答 Agent。"""

    ITERATION_LIMIT_SYNTHESIS_MESSAGE_LIMIT = 14
    ITERATION_LIMIT_SYNTHESIS_CONTEXT_LIMIT = 16000
    ITERATION_LIMIT_SYNTHESIS_ITEM_LIMIT = 2000

    def __init__(
        self,
        redis_instance: Optional[RedisInstance] = None,
        redis_cluster: Optional[RedisCluster] = None,
        progress_emitter: Optional[Any] = None,
        exclude_mcp_categories: Optional[List[Any]] = None,
        support_package_path: Optional[Path] = None,
        llm: Optional[Any] = None,
    ):
        self.redis_instance = redis_instance
        self.redis_cluster = redis_cluster
        self.exclude_mcp_categories = exclude_mcp_categories
        self.support_package_path = support_package_path
        self._emitter = progress_emitter if progress_emitter is not None else NullEmitter()
        if llm is None:
            if settings.openai_api_key is not None:
                from redis_sre_agent.core.llm_helpers import create_llm

                llm = create_llm()
            else:
                from ._compat import FakeToolCallingLLM

                llm = FakeToolCallingLLM(agent_kind="chat")
        self.llm = llm

    @staticmethod
    def _reached_iteration_limit(
        final_state: Dict[str, Any], requested_max_iterations: int
    ) -> bool:
        iteration_count = final_state.get("iteration_count")
        max_iterations = final_state.get("max_iterations", requested_max_iterations)
        return (
            isinstance(iteration_count, int)
            and isinstance(max_iterations, int)
            and iteration_count >= max_iterations
        )

    async def _synthesize_iteration_limit_response(
        self,
        *,
        query: str,
        messages: List[BaseMessage],
        tool_envelopes: List[Dict[str, Any]],
        iteration_count: int,
        max_iterations: int,
    ) -> str:
        """达到工具循环预算时，用已捕获状态请求 LLM 完成终态回答。"""

        return await synthesize_terminal_response(
            self.llm,
            config=TerminalSynthesisConfig(
                request_kind="chat_agent.iteration_limit_synthesis",
                system_prompt=(
                    "The chat workflow stopped because it reached its iteration budget "
                    "before emitting terminal assistant text. Write the best possible final "
                    "answer from the gathered conversation and tool evidence. Do not call "
                    "tools. Do not invent evidence. State remaining uncertainty explicitly."
                ),
                messages_heading="Conversation tail",
                evidence_heading="Structured tool evidence",
                no_messages_text="No non-system conversation messages were captured.",
                no_evidence_text="No structured tool result envelopes were captured.",
                failure_log_message="Chat max-iteration synthesis failed: %s",
                empty_log_message="Chat max-iteration synthesis returned empty text",
                context_limit=self.ITERATION_LIMIT_SYNTHESIS_CONTEXT_LIMIT,
                item_limit=self.ITERATION_LIMIT_SYNTHESIS_ITEM_LIMIT,
                message_item_limit=self.ITERATION_LIMIT_SYNTHESIS_ITEM_LIMIT,
                message_tail_limit=self.ITERATION_LIMIT_SYNTHESIS_MESSAGE_LIMIT,
                include_system_messages=False,
                detailed_message_headers=True,
                empty_message_text="(no text content)",
                message_omitted_unit="conversation messages",
                evidence_omitted_unit="tool result envelopes",
            ),
            messages=messages,
            tool_envelopes=tool_envelopes,
            guarded_invoke=guarded_ainvoke,
            failure_response_factory=lambda: build_deterministic_diagnostic_response(
                query,
                tool_envelopes,
                agent_kind="chat",
            ),
            logger=logger,
            human_prelude=f"Iteration budget: {iteration_count}/{max_iterations}",
        )

    def _build_workflow(
        self,
        tool_mgr: ToolManager,
        emitter: Optional[Any] = None,
    ) -> StateGraph:
        """构建 ChatAgent 的 LangGraph 风格 workflow。"""

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

        async def agent_node(state: ChatAgentState) -> Dict[str, Any]:
            """调用 LLM，让它决定下一步回答或工具调用。"""

            runtime = await ensure_runtime_tools()
            messages = list(state.get("messages") or [])
            iteration_count = state.get("iteration_count", 0)
            startup_system_prompt = state.get("startup_system_prompt") or CHAT_SYSTEM_PROMPT
            startup_prompt_initialized = state.get("startup_prompt_initialized", False)

            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=startup_system_prompt)] + messages
                startup_prompt_initialized = True

            response = await guarded_ainvoke(
                runtime["llm_with_tools"],
                messages,
                request_kind="chat_agent.agent_node",
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
                "signals_envelopes": list(state.get("signals_envelopes") or []),
            }

        async def tool_node(state: ChatAgentState) -> Dict[str, Any]:
            """执行 agent node 请求的工具，并记录结构化 evidence。"""

            runtime = await ensure_runtime_tools(state.get("toolset_generation"))
            tooldefs_by_name = runtime["tooldefs_by_name"]
            messages = list(state.get("messages") or [])
            tool_calls = list(state.get("current_tool_calls") or [])
            if not tool_calls and messages and isinstance(messages[-1], AIMessage):
                tool_calls = list(messages[-1].tool_calls or [])

            if emitter and tool_calls:
                for call in tool_calls:
                    tool_name = str(call.get("name") or "")
                    tool_args = call.get("args") if isinstance(call.get("args"), dict) else {}
                    status_msg = None
                    if tool_name and hasattr(tool_mgr, "get_status_update"):
                        status_msg = tool_mgr.get_status_update(tool_name, tool_args)
                    if status_msg and hasattr(emitter, "emit"):
                        await emitter.emit(
                            status_msg,
                            "tool_call",
                            metadata={"tool_name": tool_name, "tool_args": tool_args},
                        )

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

        def should_continue(state: ChatAgentState) -> str:
            """决定下一跳：继续调用工具，或结束。"""

            iteration_count = state.get("iteration_count", 0)
            max_iterations = state.get("max_iterations", 10)
            if iteration_count >= max_iterations:
                logger.warning("Chat agent reached max iterations (%s)", max_iterations)
                return END

            messages = state.get("messages") or []
            if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
                return "tools"
            if state.get("current_tool_calls"):
                return "tools"
            return END

        workflow = StateGraph(ChatAgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")
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
            return f"""INSTANCE CONTEXT: This query is about Redis instance:
- Instance Name: {self.redis_instance.name}
- Environment: {self.redis_instance.environment}
- Usage: {self.redis_instance.usage}
- Instance Type: {self.redis_instance.instance_type}

User Query: {query}"""
        if self.redis_cluster is not None:
            return f"""CLUSTER CONTEXT: This query is about Redis cluster:
- Cluster Name: {self.redis_cluster.name}
- Cluster ID: {self.redis_cluster.id}
- Environment: {self.redis_cluster.environment}
- Cluster Type: {self.redis_cluster.cluster_type}

User Query: {query}"""
        target_query = str(context.get("target_query") or context.get("target") or "").strip()
        if target_query and target_query != query:
            return f"Target Hint: {target_query}\n\nUser Query: {query}"
        return query

    async def process_query(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str],
        max_iterations: int = 10,
        context: Optional[Dict[str, Any]] = None,
        progress_emitter: Optional[Any] = None,
        conversation_history: Optional[List[Any]] = None,
    ) -> AgentResponse:
        """通过 ChatAgent workflow 处理一次查询。"""

        logger.info("Chat agent processing query for user %s", user_id or "<anonymous>")
        normalized_context = dict(context or {})
        thread_id = str(normalized_context.get("thread_id") or session_id or "chat")
        initial_bindings = get_target_bindings_from_context(normalized_context)
        support_package_path = self.support_package_path or normalized_context.get(
            "support_package_path"
        )
        if isinstance(support_package_path, str):
            support_package_path = Path(support_package_path)

        emitter = progress_emitter if progress_emitter is not None else self._emitter
        async with ToolManager(
            redis_instance=self.redis_instance,
            redis_cluster=self.redis_cluster,
            initial_target_bindings=initial_bindings or None,
            initial_toolset_generation=int(
                normalized_context.get("target_toolset_generation")
                or normalized_context.get("toolset_generation")
                or 0
            ),
            exclude_mcp_categories=self.exclude_mcp_categories,
            support_package_path=support_package_path,
            thread_id=thread_id,
            task_id=normalized_context.get("task_id"),
            user_id=user_id,
            graph_type="chat",
        ) as tool_mgr:
            workflow = self._build_workflow(tool_mgr, emitter)
            enhanced_query = self._enhance_query(query, normalized_context)
            initial_messages: List[BaseMessage] = [SystemMessage(content=CHAT_SYSTEM_PROMPT)]
            if conversation_history:
                initial_messages.extend(conversation_history)
            initial_messages.append(HumanMessage(content=enhanced_query))

            initial_state: ChatAgentState = {
                "messages": initial_messages,
                "session_id": session_id,
                "user_id": user_id,
                "current_tool_calls": [],
                "iteration_count": 0,
                "max_iterations": max_iterations,
                "startup_system_prompt": CHAT_SYSTEM_PROMPT,
                "startup_prompt_initialized": True,
                "toolset_generation": tool_mgr.get_toolset_generation(),
                "signals_envelopes": [],
            }
            graph_thread_id = resolve_graph_thread_id(session_id, normalized_context)
            try:
                if hasattr(emitter, "emit"):
                    await emitter.emit("Chat agent processing your question...", "agent_start")
                app = workflow.compile()
                final_state = await app.ainvoke(
                    initial_state,
                    config=build_graph_config(graph_thread_id=graph_thread_id),
                )
                tool_envelopes = list(final_state.get("signals_envelopes") or [])
                messages = list(final_state.get("messages") or [])
                response_text = extract_last_ai_response(messages, terminal_only=True)
                if response_text:
                    return AgentResponse(response=response_text, tool_envelopes=tool_envelopes)
                if self._reached_iteration_limit(final_state, max_iterations):
                    iteration_count = final_state.get("iteration_count", max_iterations)
                    state_max_iterations = final_state.get("max_iterations", max_iterations)
                    response_text = await self._synthesize_iteration_limit_response(
                        query=query,
                        messages=messages,
                        tool_envelopes=tool_envelopes,
                        iteration_count=(
                            iteration_count if isinstance(iteration_count, int) else max_iterations
                        ),
                        max_iterations=(
                            state_max_iterations
                            if isinstance(state_max_iterations, int)
                            else max_iterations
                        ),
                    )
                else:
                    response_text = "I couldn't process that query. Please try rephrasing."
                return AgentResponse(response=response_text, tool_envelopes=tool_envelopes)
            except Exception as exc:
                logger.exception("Chat agent error: %s", exc)
                return AgentResponse(response=f"Error processing query: {exc}")


_chat_agents: Dict[str, ChatAgent] = {}


def get_chat_agent(
    redis_instance: Optional[RedisInstance] = None,
    redis_cluster: Optional[RedisCluster] = None,
) -> ChatAgent:
    """按 target 缓存 ChatAgent，沿用 original 公开入口。"""

    if redis_instance and redis_cluster:
        key = f"instance:{redis_instance.id}|cluster:{redis_cluster.id}"
    elif redis_instance:
        key = f"instance:{redis_instance.id}"
    elif redis_cluster:
        key = f"cluster:{redis_cluster.id}"
    else:
        key = "__no_instance__"
    if key not in _chat_agents:
        _chat_agents[key] = ChatAgent(redis_instance=redis_instance, redis_cluster=redis_cluster)
    return _chat_agents[key]
