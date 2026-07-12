"""由跨进程测试启动的 CLI harness；不作为独立 pytest 用例收集。"""

from __future__ import annotations

import json
import importlib
import sys
from typing import Any

from click.testing import CliRunner
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from redis_sre_agent.agent.chat_agent import ChatAgent
from redis_sre_agent.agent import chat_agent as chat_agent_module
from redis_sre_agent.agent.models import AgentResponse
from redis_sre_agent.agent.router import AgentType
from redis_sre_agent.cli.main import main
from redis_sre_agent.core import instances as instances_module
from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.core.threads import ThreadManager
from redis_sre_agent.tools.manager import ToolManager


ROUTER_HISTORY: list[Any] = []
TOOL_MANAGER_INIT: dict[str, Any] = {}


class RecordingToolManager(ToolManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        TOOL_MANAGER_INIT["initial_toolset_generation"] = kwargs.get(
            "initial_toolset_generation"
        )
        TOOL_MANAGER_INIT["initial_target_bindings"] = kwargs.get("initial_target_bindings")
        super().__init__(*args, **kwargs)


class RecordingLLM:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "RecordingLLM":
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.messages = list(messages)
        return AIMessage(content="SECOND_LLM_FINAL")


async def fake_route_to_agent(
    *,
    query: str,
    context: dict[str, Any] | None = None,
    conversation_history: list[Any] | None = None,
) -> AgentType:
    ROUTER_HISTORY[:] = list(conversation_history or [])
    return AgentType.REDIS_CHAT


async def fake_get_instance_by_id(instance_id: str) -> RedisInstance | None:
    if instance_id != "inst-process-persisted":
        return None
    return RedisInstance(
        id=instance_id,
        name="Process persisted Redis",
        connection_url=SecretStr("redis://process-persisted.invalid:6379/0"),
        environment="test",
        usage="cache",
        description="Cross-process CLI test target",
    )


class ProcessFakeAgent:
    async def process_query(
        self,
        query: str,
        *,
        session_id: str,
        user_id: str | None,
        max_iterations: int,
        context: dict[str, Any],
        conversation_history: list[Any] | None = None,
    ) -> AgentResponse:
        manager = ThreadManager()
        if query == "first process question":
            await manager.update_thread_context(
                context["thread_id"],
                {
                    "instance_id": "inst-process-persisted",
                    "target_bindings": [
                        {
                            "target_handle": "tgt_process_1",
                            "target_kind": "instance",
                            "resource_id": "inst-process-persisted",
                            "display_name": "Process persisted Redis",
                            "provider_hints": {},
                            "credential_ref": None,
                        }
                    ],
                    "attached_target_handles": ["tgt_process_1"],
                    "target_toolset_generation": 7,
                },
            )
            return AgentResponse(
                response="FIRST_PROCESS_ANSWER",
                tool_envelopes=[
                    {"tool_key": "redis_process_info", "name": "info", "status": "success"}
                ],
            )

        thread = await manager.get_thread(context["thread_id"])
        previous_assistant = next(
            message for message in reversed(thread.messages) if message.role == "assistant"
        )
        trace = await manager.get_message_trace(previous_assistant.message_id or "")
        recording_llm = RecordingLLM()
        original_manager = chat_agent_module.ToolManager
        chat_agent_module.ToolManager = RecordingToolManager
        try:
            await ChatAgent(llm=recording_llm).process_query(
                query,
                session_id=session_id,
                user_id=user_id,
                max_iterations=max_iterations,
                context=context,
                conversation_history=conversation_history,
            )
        finally:
            chat_agent_module.ToolManager = original_manager
        result = {
            "session_id": session_id,
            "user_id": user_id,
            "agent_history_types": [
                message.__class__.__name__ for message in conversation_history or []
            ],
            "agent_history_content": [
                str(message.content) for message in conversation_history or []
            ],
            "router_history_types": [message.__class__.__name__ for message in ROUTER_HISTORY],
            "router_history_content": [str(message.content) for message in ROUTER_HISTORY],
            "llm_input_types": [message.__class__.__name__ for message in recording_llm.messages],
            "llm_input_content": [str(message.content) for message in recording_llm.messages],
            "tool_manager_generation": TOOL_MANAGER_INIT["initial_toolset_generation"],
            "tool_manager_binding_handles": [
                binding.target_handle
                for binding in TOOL_MANAGER_INIT["initial_target_bindings"] or []
            ],
            "context": context,
            "previous_trace": trace,
        }
        return AgentResponse(response=json.dumps(result, ensure_ascii=False))


def main_harness() -> int:
    query_module = importlib.import_module("redis_sre_agent.cli.query")
    query_module.route_to_appropriate_agent = fake_route_to_agent
    query_module.get_chat_agent = lambda **_: ProcessFakeAgent()
    instances_module.get_instance_by_id = fake_get_instance_by_id

    phase = sys.argv[1]
    if phase == "first":
        args = ["query", "first process question", "--target", "prod-cache"]
    else:
        args = ["query", "second process question", "--thread-id", sys.argv[2]]
    result = CliRunner().invoke(main, args)
    sys.stdout.write(result.output)
    if result.exception is not None:
        sys.stderr.write(f"{type(result.exception).__name__}: {result.exception}\n")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main_harness())
