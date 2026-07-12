"""使用真实系统 Redis 验证两次独立 CLI 进程的 Thread 连续性。"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from redis_sre_agent.core.keys import RedisKeys
from redis_sre_agent.core.redis import get_redis_client
from redis_sre_agent.core.threads import ThreadManager


ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).with_name("cli_thread_process_harness.py")


def _run_process(*args: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(HARNESS), *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


async def _redis_available() -> bool:
    client = get_redis_client()
    try:
        return bool(await client.ping())
    except Exception:
        return False
    finally:
        await client.aclose()


@pytest.mark.integration
def test_cli_thread_context_history_and_trace_survive_process_boundary() -> None:
    if not asyncio.run(_redis_available()):
        pytest.skip("系统管理 Redis 当前不可用")

    first = _run_process("first")
    thread_id = first["thread_id"]
    second = _run_process("second", thread_id)
    observed = json.loads(second["response"])

    assert second["thread_id"] == thread_id
    assert observed["agent_history_types"] == ["HumanMessage", "AIMessage"]
    assert observed["agent_history_content"] == [
        "first process question",
        "FIRST_PROCESS_ANSWER",
    ]
    assert observed["router_history_types"] == ["HumanMessage", "AIMessage"]
    assert observed["router_history_content"] == observed["agent_history_content"]
    llm_pairs = list(zip(observed["llm_input_types"], observed["llm_input_content"]))
    assert ("HumanMessage", "first process question") in llm_pairs
    assert ("AIMessage", "FIRST_PROCESS_ANSWER") in llm_pairs
    assert observed["context"]["instance_id"] == "inst-process-persisted"
    assert observed["context"]["target_bindings"][0]["target_handle"] == "tgt_process_1"
    assert observed["context"]["target_toolset_generation"] == 7
    assert observed["tool_manager_generation"] == 7
    assert observed["tool_manager_binding_handles"] == ["tgt_process_1"]
    assert observed["previous_trace"]["tool_envelopes"][0]["name"] == "info"

    async def verify_and_cleanup() -> None:
        client = get_redis_client()
        manager = ThreadManager(redis_client=client)
        try:
            thread = await manager.get_thread(thread_id)
            assert thread is not None
            assert [message.role for message in thread.messages] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]
            keys = RedisKeys.all_thread_keys(thread_id)
            assert await client.type(keys["messages"]) == b"list"
            assert await client.type(keys["context"]) == b"hash"
            assert await client.type(keys["metadata"]) == b"hash"
            trace_keys = [
                RedisKeys.message_decision_trace(message.message_id or "")
                for message in thread.messages
                if message.role == "assistant"
            ]
            await client.delete(*keys.values(), *trace_keys)
        finally:
            await client.aclose()

    asyncio.run(verify_and_cleanup())
