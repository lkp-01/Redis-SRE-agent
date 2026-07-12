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

    messages: List[BaseMessage] # 消息历史
    session_id: str
    user_id: Optional[str]
    current_tool_calls: List[Dict[str, Any]] # 这一轮LLM要调用的函数
    iteration_count: int # 这一轮 思考-工具 循环的次数
    max_iterations: int # 最大循环次数
    startup_system_prompt: Optional[str]
    startup_prompt_initialized: NotRequired[bool] # 标记初始提示词插入messages头部成功与否
    instance_context: Optional[Dict[str, Any]] # 当前诊断实例/cluster的元数据，配置、环境等
    toolset_generation: NotRequired[int] # toolmanager中工具集的版本号
    signals_envelopes: List[Dict[str, Any]] # 存储所有工具调用的入参、出参及执行状态，诊断报告的核心


class SRELangGraphAgent:
    """基于 StateGraph 的 Redis 深度诊断 Agent。"""

    def __init__(
        self,
        redis_instance: Optional[RedisInstance] = None,
        redis_cluster: Optional[RedisCluster] = None,
        progress_emitter: Optional[Any] = None, #进度发射器，向前端或平台推送Agent状态更新
        exclude_mcp_categories: Optional[List[Any]] = None, # 需要排除的MCP工具分类列表
        support_package_path: Optional[Path] = None, # 离线支持包路径，分析历史诊断快照
        llm: Optional[Any] = None, #自定义底层大语言模型实例
        **_: Any, #接受并忽略其它不关心的多余关键字参数
    ):
        self.settings = settings # 全局配置对象挂载到实例属性上
        self.redis_instance = redis_instance
        self.redis_cluster = redis_cluster
        self.exclude_mcp_categories = exclude_mcp_categories
        self.support_package_path = support_package_path
        self._progress_emitter = (
            progress_emitter if progress_emitter is not None else NullEmitter()
        )
        if llm is None: #优先使用已配置的真实模型；无 key 时保留测试 fallback
            if settings.openai_api_key is not None:
                from redis_sre_agent.core.llm_helpers import create_llm

                llm = create_llm()
            else:
                from ._compat import FakeToolCallingLLM

                llm = FakeToolCallingLLM(agent_kind="triage")
        self.llm = llm
        self.llm_with_tools = self.llm
        # 如果后续完全不需要绑定任何工具，它就直接用原地址；
        # 而一旦需要绑定工具，工作流内部会动态把新地址分配给运行时去消费。

    def _build_workflow(
        self,
        tool_mgr: ToolManager, ##
        target_instance: Optional[RedisInstance] = None,
    ) -> StateGraph:
        # 返回一个构建完毕的 LangGraph 状态图对象
        """构建 SRE triage workflow。"""

        # 思考的第一圈循环，ToolManager解析的工具会把“运行环境”(工具版本号、工具名等等)生成对象存入这个字典中
        # 后续循环如果版本没变、工具数量没变，继续用这个工具字典
        # 称之为闭包缓存（只存在于这一次workflow中）
        runtime_tools_by_generation: Dict[tuple[int, int], Dict[str, Any]] = {}

        # 确认当下是最新的工具清单然后包装成大模型认识的格式
        async def ensure_runtime_tools(
                requested_generation: Optional[int] = None,  # 允许指定获取某一特定版本的工具集
        ) -> Dict[str, Any]:

            # 从 tool_mgr 中动态获取当前系统最新的工具集版本代号
            current_generation = tool_mgr.get_toolset_generation()
            # 如果指定了版本，就用指定的；否则默认采用最新版本代号
            generation = requested_generation or current_generation

            # 如果请求的版本和当前最新版本不一致，强制同步为最新版本
            if generation != current_generation:
                generation = current_generation

            # 兼容性处理：尝试调用不同版本的 API 获取格式化好的 LLM 工具定义，并限制单次最多加载 64 个工具
            try:
                tooldefs = tool_mgr.get_tools_for_llm(max_tools=64)
            except TypeError:
                tooldefs = tool_mgr.get_tools_for_llm()
            except AttributeError:
                tooldefs = tool_mgr.get_tools()

            # 生成当前工具快照的唯一 Key：(版本号, 工具总数)
            cache_key = (generation, len(tooldefs))
            # 检查闭包缓存字典，如果命中缓存则直接返回，不再向下执行绑定逻辑
            cached = runtime_tools_by_generation.get(cache_key)
            if cached is not None:
                return cached

            # 异步构建工具适配器，将底层原子工具包装成大模型所需的 JSON Schema 标准格式
            adapters = await build_adapters_for_tooldefs(tool_mgr, tooldefs)
            llm_tools = adapters or tooldefs

            # 动态绑定：如果底层 LLM 支持 bind_tools 方法，则将工具动态“灌入”大模型中，生成带工具的新模型实例
            llm_with_tools = (
                self.llm.bind_tools(llm_tools) if hasattr(self.llm, "bind_tools") else self.llm
            )

            # 组装当前版本的运行时上下文快照
            runtime = {
                "generation": generation,
                "tooldefs_by_name": {tool.name: tool for tool in tooldefs},  # 构建工具名到工具定义的映射字典，便于 tool_node 快速检索
                "llm_with_tools": llm_with_tools,  # 已经绑定好最新工具的模型对象
            }
            # 将这个快照写入闭包缓存中
            runtime_tools_by_generation[cache_key] = runtime
            return runtime

        async def agent_node(state: AgentState) -> Dict[str, Any]:
            """调用 LLM 决定下一批工具或最终回答。"""

            # 确保获取当前最新的带工具模型快照
            runtime = await ensure_runtime_tools()
            # 从全局状态中拷贝一份消息历史，防止直接修改状态对象
            messages = list(state.get("messages") or [])
            # 获取当前的迭代循环次数
            iteration_count = state.get("iteration_count", 0)
            # 获取系统初始提示词，若无则使用兜底的 SRE_SYSTEM_PROMPT
            startup_system_prompt = state.get("startup_system_prompt") or SRE_SYSTEM_PROMPT
            startup_prompt_initialized = state.get("startup_prompt_initialized", False)

            # 初始化拦截：若历史消息为空，或队列头部没有系统提示词，说明是首轮对话
            if not messages or not isinstance(messages[0], SystemMessage):
                # 将系统提示词强行插入历史消息队列的队首
                messages = [SystemMessage(content=startup_system_prompt)] + messages
                # 标记初始化成功
                startup_prompt_initialized = True

            # 异步调用带工具的模型，该安全函数会拦截并包装网络或接口异常
            response = await guarded_ainvoke(
                runtime["llm_with_tools"],
                messages,
                request_kind="langgraph_agent.agent_node",
            )

            # 容错兜底：若模型的响应不是标准的 AIMessage，则提取其内容重新包装成标准对象
            if not isinstance(response, AIMessage):
                response = AIMessage(
                    content=getattr(response, "content", response),
                    tool_calls=list(getattr(response, "tool_calls", []) or []),
                )

            # 返回更新字典，LangGraph 会将这些字段自动 Merge 合并回全局 AgentState 中
            return {
                "messages": list(state.get("messages") or []) + [response],  # 把大模型这一轮的碎碎念追加到消息历史中
                "iteration_count": iteration_count + 1,  # 迭代次数加 1
                "startup_system_prompt": startup_system_prompt,
                "startup_prompt_initialized": startup_prompt_initialized,
                "toolset_generation": runtime["generation"],  # 记录大模型看这批工具时的版本号
                "current_tool_calls": list(response.tool_calls or []),  # 提取出大模型这一轮想要调用的工具列表
                "instance_context": state.get("instance_context"),
                "signals_envelopes": list(state.get("signals_envelopes") or []),
            }

        async def tool_node(state: AgentState) -> Dict[str, Any]:
            """执行工具并把 ToolMessage / ResultEnvelope 写回状态。"""

            # 提取大模型生成工具时对应的工具版本，确保环境一致
            runtime = await ensure_runtime_tools(state.get("toolset_generation"))
            tooldefs_by_name = runtime["tooldefs_by_name"]
            messages = list(state.get("messages") or [])
            # 从状态中获取当前等待执行的工具调用请求
            tool_calls = list(state.get("current_tool_calls") or [])

            # 容错：如果当前执行队列为空，但上一条消息是 AIMessage，说明大模型的调用藏在消息末尾，重新抓取出来
            if not tool_calls and messages and isinstance(messages[-1], AIMessage):
                tool_calls = list(messages[-1].tool_calls or [])
            # 如果确认没有任何工具需要调用，直接跳过退出
            if not tool_calls:
                return {}

            # 进度汇报循环：遍历所有即将调用的工具，通知前端当前 Agent 正在干什么
            for call in tool_calls:
                tool_name = str(call.get("name") or "")
                tool_args = call.get("args") if isinstance(call.get("args"), dict) else {}
                status_msg = None
                # 如果 tool_mgr 实现了状态信息生成方法，获取当前工具的语义化进度描述（如：“正在检查内存配置...”）
                if tool_name and hasattr(tool_mgr, "get_status_update"):
                    status_msg = tool_mgr.get_status_update(tool_name, tool_args)
                # 如果获取到了描述且发射器可用，异步将其向外推送给用户端
                if status_msg and hasattr(self._progress_emitter, "emit"):
                    await self._progress_emitter.emit(status_msg, "agent_reflection")

            # 安全网闸调用：异步并行执行这一批所有的工具调用，并经过内置的安全策略拦截保护
            tool_messages = await execute_tool_calls_with_gate(
                tool_manager=tool_mgr,
                tool_calls=tool_calls,
            )

            # 报告信封封装：把工具的入参、出参和状态打包在一起，这是后续生成诊断报告的“原材料”
            envelopes = list(state.get("signals_envelopes") or [])
            for index, call in enumerate(tool_calls):
                tool_name = str(call.get("name") or f"tool_{index + 1}")
                tool_args = call.get("args") if isinstance(call.get("args"), dict) else {}
                tool_message = tool_messages[index] if index < len(tool_messages) else None
                if isinstance(tool_message, ToolMessage):
                    # 将工具的真实回执（ToolMessage）组装成标准信封并追加到队列中
                    envelopes.append(
                        build_result_envelope(
                            tool_name,
                            tool_args,
                            tool_message,
                            tooldefs_by_name,
                        )
                    )

            # 动态目标发现：如果排查过程中发现了新节点，触发后备函数将其动态挂载到可用的工具箱中
            await self._attach_target_tools_from_resolution(tool_mgr, envelopes)

            # 将执行完的结果 Merge 回全局状态，并清空当前待执行队列，更新工具版本号
            return {
                "messages": messages + tool_messages,  # 把 ToolMessage 结果拼接到消息树中，供大模型下一轮看
                "current_tool_calls": [],  # 清空工具队列，表示这轮工具做完了
                "toolset_generation": tool_mgr.get_toolset_generation(),  # 标记最新的工具版本
                "signals_envelopes": envelopes,  # 存入最新的排查证据信封
            }

        async def reasoning_node(state: AgentState) -> Dict[str, Any]:
            """从已收集 evidence 生成最终回答。"""

            messages = list(state.get("messages") or [])
            envelopes = list(state.get("signals_envelopes") or [])
            query = ""

            # 从后往前遍历历史消息，精准定位用户最初输入的那个 Human 原始问题
            for message in reversed(messages):
                if isinstance(message, HumanMessage):
                    query = str(message.content)
                    break

            # 调用一个确定性的文本模板函数，根据用户的原始问题和所有的工具调用信封（envelopes）
            # 严格依据证据生成一份逻辑清晰、无幻觉的 SRE 诊断报告，包装成 AIMessage
            response = AIMessage(
                content=build_deterministic_diagnostic_response(
                    query,
                    envelopes,
                    agent_kind="triage",
                )
            )

            # 返回最新的状态增量
            return {"messages": messages + [response], "signals_envelopes": envelopes}

        def should_continue(state: AgentState) -> str:
            """决定继续工具循环、进入 reasoning，或结束。"""

            iteration_count = state.get("iteration_count", 0)
            # 获取允许的最大迭代次数（防死循环）
            max_iterations = state.get("max_iterations", self.settings.max_iterations)
            messages = state.get("messages") or []

            # 情况 1：如果迭代次数已经达到了上限，强行拦截，强制去生成报告，不能再调工具了
            if iteration_count >= max_iterations:
                logger.warning("SRE agent reached max iterations (%s)", max_iterations)
                return "reasoning"

            # 情况 2：若模型最后一条回复包含待处理的工具调用，或者队列里还有活没干完，路由到 tools 节点执行工具
            if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
                return "tools"
            if state.get("current_tool_calls"):
                return "tools"

            # 情况 3：如果有过工具执行记录，且这一轮大模型没有提出新的调用申请，说明排障完毕，去推理节点做总结
            if any(isinstance(message, ToolMessage) for message in messages):
                return "reasoning"

            # 情况 4：如果什么都没干模型就直接给出了回答（比如用户问了个纯概念问题），直接结束对话流程
            return END

        # 1. 初始化状态图，指定这个图所遵循的数据 Schema（即 AgentState 字典结构）
        workflow = StateGraph(AgentState)

        # 2. 将上面定义的三个内部异步函数作为“车间节点（Nodes）”注册到图中
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("reasoning", reasoning_node)

        # 3. 设定图的唯一主入口：机器一旦通电启动，首先进入 "agent" 车间
        workflow.set_entry_point("agent")

        # 4. 在 "agent" 车间出口挂上红绿灯（should_continue 路由规则）
        # 告诉系统：从 agent 出来后，根据 should_continue 的返回值，分别去往 tools、reasoning 房间或直接挂断(END)
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "reasoning": "reasoning", END: END},
        )

        # 5. 铺设单向输送轨道：从 "tools" 房间出来后，必须无条件重新回到 "agent" 房间让大模型看结果
        workflow.add_edge("tools", "agent")
        # 6. 铺设单向输送轨道：从 "reasoning" 房间出来后，说明报告生成完毕，无条件走向终点（END）
        workflow.add_edge("reasoning", END)

        # 7. 返回这个已经规划完毕、各轨道连接无误的完整状态图对象，等待后续编译
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

    # 提示词增强函数：第一个是显示绑定了具体单例Redis实例，第二个是整个redis集群，第三个是无显示绑定、从隐式字典提取目标提示词
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

    #负责动态组装工具、初始化状态机、编译并运行 LangGraph 流程
    async def _process_query(
            self,
            query: str,  # 用户提出的原始问题
            session_id: str,  # 当前会话ID，用于追踪
            user_id: Optional[str],  # 用户ID
            max_iterations: int,  # Agent 思考-调用工具 的最大循环次数限制
            context: Optional[Dict[str, Any]] = None,  # 包含集群/实例信息的上下文
            conversation_history: Optional[List[BaseMessage]] = None,  # 历史聊天记录
            progress_emitter: Optional[Any] = None,  # 进度发射器，用于向前端推送执行状态
    ) -> AgentResponse:  # 返回结构化的诊断响应

        # 1. 准备和规范化上下文数据
        normalized_context = dict(context or {})  # 确保 context 是个字典，避免 None 导致报错

        # 如果外部传入了新的进度发射器，覆盖实例默认的发射器
        if progress_emitter is not None:
            self._progress_emitter = progress_emitter

        # 确定当前的线程 ID，优先用 context 里的，其次用 session_id，都没有就用默认值 "triage"
        thread_id = str(normalized_context.get("thread_id") or session_id or "triage")

        # 从上下文中提取绑定的目标（比如特定的 Redis 节点或资源）
        initial_bindings = get_target_bindings_from_context(normalized_context)

        # 获取离线诊断包（support package）的路径，优先用实例属性，其次查上下文
        support_package_path = self.support_package_path or normalized_context.get(
            "support_package_path"
        )
        # 如果路径是字符串，将其转化为 pathlib.Path 对象以便后续处理
        if isinstance(support_package_path, str):
            support_package_path = Path(support_package_path)

        # 2. 初始化工具管理器 (ToolManager)
        # 这是一个异步上下文管理器，确保执行完后会正确释放资源
        async with ToolManager(
                redis_instance=self.redis_instance,  # 注入当前 Redis 实例
                redis_cluster=self.redis_cluster,  # 注入当前 Redis 集群
                initial_target_bindings=initial_bindings or None,  # 注入初始绑定的目标
                exclude_mcp_categories=self.exclude_mcp_categories,  # 过滤掉不需要的工具分类
                support_package_path=support_package_path,  # 注入离线诊断包路径（如果有）
                thread_id=thread_id,
                task_id=normalized_context.get("task_id"),
                user_id=user_id,
                graph_type="redis_triage",  # 标记这是排查图谱
        ) as tool_mgr:

            # 3. 构建工作流与增强提示词
            # 调用前面定义的方法，把 ToolManager 和 实例绑在一起，生成 LangGraph 的图结构
            self.workflow = self._build_workflow(tool_mgr, self.redis_instance)

            # 根据当前绑定的实例/集群，给用户的原始 query 加上重要的上下文前缀
            enhanced_query = self._enhance_query(query, normalized_context)

            # 4. 初始化对话历史 (Message List)
            # 第一条消息强制注入系统设定 (SRE_SYSTEM_PROMPT)
            initial_messages: List[BaseMessage] = [SystemMessage(content=SRE_SYSTEM_PROMPT)]
            # 如果有之前的对话历史，把它们接在系统提示词后面
            if conversation_history:
                initial_messages.extend(conversation_history)
            # 最后一条消息放当前用户提出的问题（已被增强过）
            initial_messages.append(HumanMessage(content=enhanced_query))

            # 5. 初始化 LangGraph 全局状态
            initial_state: AgentState = {
                "messages": initial_messages,  # 携带历史记录和本次问题的消息列表
                "session_id": session_id,
                "user_id": user_id,
                "current_tool_calls": [],  # 当前需要调用的工具队列，初始为空
                "iteration_count": 0,  # 思考迭代次数，初始为0
                "max_iterations": max_iterations,  # 限制最大循环防死循环
                "startup_system_prompt": SRE_SYSTEM_PROMPT,
                "startup_prompt_initialized": True,  # 标记提示词已经就绪
                "instance_context": normalized_context or None,  # 携带环境变量
                "toolset_generation": tool_mgr.get_toolset_generation(),  # 记录当前的工具版本
                "signals_envelopes": [],  # 收集执行结果的信封队列，初始为空
            }

            # 解析出一个给 LangGraph 底层用的 thread_id，用于 Checkpoint 机制（状态保存与恢复）
            graph_thread_id = resolve_graph_thread_id(session_id, normalized_context)

            # 6. 编译并运行图 (Compile & Execute)
            # 将图结构编译成可执行的 application
            app = self.workflow.compile()
            self.app = app

            # 触发图谱异步执行 (ainvoke)，大模型会在这里开始循环推理和调用工具
            final_state = await app.ainvoke(
                initial_state,
                config=build_graph_config(  # 传递图谱运行配置
                    graph_thread_id=graph_thread_id,
                    recursion_limit=getattr(self.settings, "recursion_limit", 100),  # 防止图的节点无限制递归
                ),
            )

            # 7. 提取结果并格式化返回
            # 获取运行结束后所有工具执行的记录
            tool_envelopes = list(final_state.get("signals_envelopes") or [])
            messages = list(final_state.get("messages") or [])

            # 尝试从消息记录的末尾提取大模型的最后一次发言（只提取纯文本回答）
            response_text = extract_last_ai_response(messages, terminal_only=True)

            # 如果大模型由于某些原因（比如达到最大迭代次数）没有给出总结陈词
            if not response_text:
                # 触发后备机制：用已收集到的工具数据，通过一个确定性的函数拼装一份报告
                response_text = build_deterministic_diagnostic_response(
                    query,
                    tool_envelopes,
                    agent_kind="triage",
                )

                # 返回标准的格式化响应：包含文本回答和具体的工具执行记录
            return AgentResponse(response=response_text, tool_envelopes=tool_envelopes)

    # 对外接口
    async def process_query(
            self,
            query: str,  # 用户提出的问题
            session_id: str,  # 会话 ID
            user_id: Optional[str],  # 用户 ID
            max_iterations: int = settings.max_iterations,  # 从全局配置读取默认的最大循环次数
            context: Optional[Dict[str, Any]] = None,  # 上下文环境变量
            conversation_history: Optional[List[BaseMessage]] = None,  # 聊天历史
            progress_emitter: Optional[Any] = None,  # 进度发射器
    ) -> AgentResponse:
        """处理一次 SRE 查询。"""

        # 打印信息级日志：记录当前是谁（哪个用户）发起了查询请求
        logger.info("Processing SRE query for user %s", user_id or "<anonymous>")

        try:
            # 安全地调用内部的真正执行逻辑，原封不动地传递所有参数
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
            # 捕获在整个 LangGraph 组装、运行、网络请求期间发生的【任何】崩溃或异常
            # 记录详细的异常堆栈信息，方便排查问题
            logger.exception("SRE agent error: %s", exc)

            # 返回一个降级的 AgentResponse 对象，告知调用方发生了错误，保证系统不会直接崩溃退出
            return AgentResponse(response=f"Error processing query: {exc}")

    # 对外公开对口的诊断主入口：对底层逻辑进行了一层通用的异常拦截保护
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
