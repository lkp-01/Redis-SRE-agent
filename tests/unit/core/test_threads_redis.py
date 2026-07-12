"""ThreadManager Redis 持久化契约。"""

from __future__ import annotations

import re

import pytest

from redis_sre_agent.core.keys import RedisKeys
from redis_sre_agent.core.threads import ThreadManager
from tests.support.fake_redis import FakeRedis


ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


@pytest.mark.asyncio
async def test_thread_and_trace_survive_new_manager_instance() -> None:
    redis = FakeRedis()
    first = ThreadManager(redis_client=redis)
    thread_id = await first.create_thread(
        user_id="u1",
        session_id="cli:session-1",
        initial_context={
            "instance_id": "inst-1",
            "target_bindings": [{"target_handle": "tgt-1", "kind": "instance"}],
            "target_toolset_generation": 3,
        },
        tags=["cli"],
    )
    await first.append_messages(
        thread_id,
        [
            {"role": "user", "content": "first question"},
            {
                "role": "assistant",
                "content": "first answer",
                "metadata": {"message_id": "01J00000000000000000000000"},
            },
        ],
    )
    await first.set_message_trace(
        message_id="01J00000000000000000000000",
        tool_envelopes=[{"name": "info", "status": "success", "data": {"used": 1}}],
    )

    second = ThreadManager(redis_client=redis)
    restored = await second.get_thread(thread_id)
    trace = await second.get_message_trace("01J00000000000000000000000")

    assert ULID_RE.fullmatch(thread_id)
    assert restored is not None
    assert [message.role for message in restored.messages] == ["user", "assistant"]
    assert ULID_RE.fullmatch(restored.messages[0].message_id or "")
    assert restored.context["instance_id"] == "inst-1"
    assert restored.context["target_bindings"][0]["target_handle"] == "tgt-1"
    assert restored.context["target_toolset_generation"] == 3
    assert trace is not None and trace["tool_envelopes"][0]["name"] == "info"


@pytest.mark.asyncio
async def test_thread_uses_original_redis_types_keys_and_ttls() -> None:
    redis = FakeRedis()
    manager = ThreadManager(redis_client=redis)
    thread_id = await manager.create_thread(initial_context={"nested": {"items": [1, 2]}})
    await manager.append_messages(thread_id, [{"role": "user", "content": "hello"}])
    await manager.set_message_trace(message_id="01J00000000000000000000001", tool_envelopes=[])
    keys = RedisKeys.all_thread_keys(thread_id)
    trace_key = RedisKeys.message_decision_trace("01J00000000000000000000001")

    assert await redis.type(keys["messages"]) == b"list"
    assert await redis.type(keys["context"]) == b"hash"
    assert await redis.type(keys["metadata"]) == b"hash"
    assert await redis.type(trace_key) == b"string"
    assert await redis.ttl(keys["messages"]) == 24 * 60 * 60
    assert await redis.ttl(keys["context"]) == 24 * 60 * 60
    assert await redis.ttl(keys["metadata"]) == 24 * 60 * 60
    assert await redis.ttl(trace_key) == 7 * 24 * 60 * 60


def test_threads_module_has_no_process_global_backend() -> None:
    from redis_sre_agent.core import threads

    assert not hasattr(threads, "_THREADS")
    assert not hasattr(threads, "_MESSAGE_TRACES")
