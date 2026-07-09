"""Top-level `query` CLI command."""

# 本文件承载 `redis-sre-agent query` 的生命周期：解析目标 scope，创建或恢复
# Thread，调用 router 和 Agent，再把回答与工具 evidence 写回 Thread。

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional
from uuid import uuid4

import click
from langchain_core.messages import AIMessage, HumanMessage

from redis_sre_agent.agent.chat_agent import get_chat_agent
from redis_sre_agent.agent.langgraph_agent import get_sre_agent
from redis_sre_agent.agent.router import AgentType, route_to_appropriate_agent
from redis_sre_agent.cli.logging_utils import log_cli_exception
from redis_sre_agent.core.config import settings
from redis_sre_agent.core.threads import ThreadManager

logger = logging.getLogger(__name__)


@click.command() #意思是query现在是可以直接在终端运行的cli命令
#下面全是cli命令的生成。在终端执行时，这个值直接跟在对应命令后面。其中argument是必选(位置随便)，其余是可选(有-，--就行)
@click.argument("query")
@click.option(
    "--redis-instance-id",
    "-r",
    "--instance-id",
    "redis_instance_id",
    help="Redis instance ID to investigate",
)
@click.option("--redis-cluster-id", "-c", help="Redis cluster ID to investigate")
@click.option("--support-package-id", "-p", help="Support package ID to analyze")
@click.option("--thread-id", "-t", help="Thread ID to continue an existing conversation")
@click.option("--user-id", help="User ID to scope thread ownership and memory retrieval")
@click.option("--target", help="Natural-language target hint, such as 'prod checkout cache'")
@click.option(
    "--agent",
    "-a",
    type=click.Choice(["auto", "triage", "chat", "knowledge"], case_sensitive=False),
    default="auto",
    help="Agent to use (default: auto-select based on query)",
)
def query(
    query: str,
    redis_instance_id: Optional[str],
    redis_cluster_id: Optional[str],
    support_package_id: Optional[str],
    thread_id: Optional[str],
    user_id: Optional[str],
    target: Optional[str],
    agent: str,
) -> None:
    """Execute an agent query."""

    async def _query() -> None:
        from redis_sre_agent.core import clusters as clusters_module
        from redis_sre_agent.core import instances as instances_module

        if redis_instance_id and redis_cluster_id:
            raise click.ClickException(
                "Please provide only one of --redis-instance-id or --redis-cluster-id"
            )

        thread_manager = ThreadManager() ##

        instance = None
        if redis_instance_id:
            instance = await instances_module.get_instance_by_id(redis_instance_id)
            if not instance:
                raise click.ClickException(f"Instance not found: {redis_instance_id}")

        cluster = None
        if redis_cluster_id:
            cluster = await clusters_module.get_cluster_by_id(redis_cluster_id)##
            if not cluster:
                raise click.ClickException(f"Cluster not found: {redis_cluster_id}")

        active_thread_id = thread_id
        active_session_id = thread_id
        resolved_user_id = user_id
        conversation_history = []
        base_context = {}

        if thread_id:# 如果用户在命令行指定了 --thread-id，说明是要“恢复并继续”之前的对话
            thread = await thread_manager.get_thread(thread_id)
            if not thread:
                raise click.ClickException(f"Thread not found: {thread_id}")
            resolved_user_id = user_id or thread.metadata.user_id
            active_session_id = thread.metadata.session_id or thread_id
            base_context.update(thread.context or {})
            for message in thread.messages:# 遍历该会话历史中存储的所有聊天消息  ##
                if message.role == "user":# 如果是用户发的消息，将其转换为 LangChain 标准的 HumanMessage 对象，并追加到历史列表
                    conversation_history.append(HumanMessage(content=message.content))
                elif message.role == "assistant":# 如果是 AI 助手回的消息，将其转换为 LangChain 标准的 AIMessage 对象，并追加到历史列表
                    conversation_history.append(AIMessage(content=message.content))
                    # 兜底容错逻辑 1：如果用户本次运行命令行时没有传 --redis-instance-id 和 --redis-cluster-id，
                    # 并且历史 Thread 上下文中存有 instance_id，则自动去数据库查出该 Redis 实例对象并恢复
                    if not instance and not cluster and thread.context.get("instance_id"):
                        instance = await instances_module.get_instance_by_id(thread.context["instance_id"])##
                    if not instance and not cluster and thread.context.get("cluster_id"):
                        cluster = await clusters_module.get_cluster_by_id(thread.context["cluster_id"])
        else:# 如果用户没有指定 --thread-id，说明这是一个全新的提问会话
            active_session_id = f"cli:{uuid4()}" #创建一个全新session_id
            initial_context = {} #新字典，放上下文
            # 如果用户在命令行传入了具体的 Redis 实例/集群/故障包/目标提示 ID，记录到初始上下文中
            if redis_instance_id:
                initial_context["instance_id"] = redis_instance_id
            if redis_cluster_id:
                initial_context["cluster_id"] = redis_cluster_id
            if support_package_id:
                initial_context["support_package_id"] = support_package_id
            if target:
                initial_context["target_query"] = target
            # 异步调用存储管理器，在数据库中正式创建一个全新的 Thread 记录，
            # 并给它打上 ["cli"] 标签，同时拿到系统新生成的 active_thread_id
            active_thread_id = await thread_manager.create_thread(
                user_id=resolved_user_id,
                session_id=active_session_id,
                initial_context=initial_context,
                tags=["cli"],
            )
            await thread_manager.update_thread_subject(active_thread_id, query)
            base_context.update(initial_context)

        # 浅拷贝 base_context 字典，创建一个全新的 context 字典，防止后续的修改污染到原有的 base_context
        context = dict(base_context)
        # 将当前处于激活状态的线程 ID（无论是新生成的还是恢复的）存入 context 中
        context["thread_id"] = active_thread_id
        # 如果前面成功获取到了单机版 Redis 实例对象/集群···，则将其 ID 注入上下文中
        if instance:
            context["instance_id"] = instance.id
        elif cluster:
            context["cluster_id"] = cluster.id
        if support_package_id:
            context["support_package_id"] = support_package_id
        if target:
            context["target_query"] = target

        routing_keys = {# 定义一个set，里面包含了智能路由（Router）关心的特征 Key 列表
            "instance_id",
            "cluster_id",
            "support_package_id",
            "support_package_path",
            "attached_target_handles",
            "target_query",
        }
        routing_context = { #context中的key存在与=于routing_keys的才加入到这个字典里
            key: value
            for key, value in context.items()
            if key in routing_keys
        }

        agent_choice_map = { ## 将命令行输入的字符串参数映射为底层代码定义的 Agent 枚举类型
            "triage": AgentType.REDIS_TRIAGE,
            "chat": AgentType.REDIS_CHAT,
            "knowledge": AgentType.REDIS_CHAT,
        }
        # 如果用户在命令行明确指定了特定的 Agent
        if agent != "auto":
            agent_type = agent_choice_map[agent.lower()]
        else:#异步调用路由模块，让 LLM 根据当前用户的提问、路由上下文以及历史聊天记录，智能判断应该分发给哪个 Agent
            agent_type = await route_to_appropriate_agent( ##
                query=query,
                context=routing_context,
                conversation_history=conversation_history or None,
            )

        # 要么诊断，要么闲聊
        if agent_type == AgentType.REDIS_TRIAGE:##
            selected_agent = get_sre_agent(redis_instance=instance, redis_cluster=cluster)
        else:
            selected_agent = get_chat_agent(redis_instance=instance, redis_cluster=cluster)

        await thread_manager.append_messages( # 用户本次问题追加到该线程的历史消息里
            active_thread_id,
            [{"role": "user", "content": query}],
        )

        agent_response = await selected_agent.process_query(
            query,
            session_id=active_session_id or active_thread_id or "cli",
            user_id=resolved_user_id,
            max_iterations=settings.max_iterations, ##
            context=context,
            conversation_history=conversation_history or None,
        )

        assistant_message_id = str(uuid4())
        if agent_response.tool_envelopes:# 链路追踪追踪（Trace）：如果 Agent 在思考过程中调用了任何底层工具并返回了证据包（tool_envelopes）
            await thread_manager.set_message_trace(
                message_id=assistant_message_id,
                tool_envelopes=agent_response.tool_envelopes,
            )

        await thread_manager.append_messages( # 将 AI 助手的最终文本回答，连同刚刚生成的 message_id 扩展元数据，正式追加持久化到线程历史消息库中
            active_thread_id,
            [
                {
                    "role": "assistant",
                    "content": agent_response.response,
                    "metadata": {"message_id": assistant_message_id},
                },
            ],
        )

        payload = agent_response.model_dump(mode="json")# 将 Pydantic 模型对象转为标准的 Python 纯 JSON 序列化字典数据
        payload["thread_id"] = active_thread_id #返回给终端的数据载荷中补上当前的线程 ID，方便终端用户或调用者后续追踪
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str)) #到屏幕

    #执行上面定义好的异步内部函数 _query()
    try:
        asyncio.run(_query())
    except click.ClickException:
        raise
    except Exception as exc:
        log_cli_exception(__name__, "query CLI command failed", exc)
        raise click.ClickException(str(exc)) from exc
