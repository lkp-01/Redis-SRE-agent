"""使用系统 Redis 持久化 CLI/Agent 的多轮 Thread。"""

from __future__ import annotations

import json
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from redis.asyncio import Redis

from redis_sre_agent.core.keys import RedisKeys
from redis_sre_agent.core.redis import get_redis_client

THREAD_TTL_SECONDS = 24 * 60 * 60
MESSAGE_TRACE_TTL_SECONDS = 7 * 24 * 60 * 60
_CROCKFORD_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_ulid() -> str:
    """生成 original 使用的 26 位、时间有序 ULID 字符串形状。"""

    value = ((int(time.time() * 1000) & ((1 << 48) - 1)) << 80) | secrets.randbits(80)
    encoded = ["0"] * 26
    for index in range(25, -1, -1):
        encoded[index] = _CROCKFORD_BASE32[value & 31]
        value >>= 5
    return "".join(encoded)


def _decode(value: Any) -> Any:
    return value.decode() if isinstance(value, bytes) else value


def _serialize_context_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _deserialize_context_value(value: Any) -> Any:
    text = _decode(value)
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text


class Message(BaseModel):
    """Thread 中一条可见消息，message_id 使用 ULID。"""

    message_id: Optional[str] = Field(default=None)
    role: str = Field(default="user")
    content: str
    metadata: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
        if self.message_id is None:
            object.__setattr__(self, "message_id", _new_ulid())


class ThreadMetadata(BaseModel):
    """Thread 元数据，字段形状沿用 original。"""

    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: int = 0
    tags: List[str] = Field(default_factory=list)
    subject: Optional[str] = None


class Thread(BaseModel):
    """完整多轮会话状态。"""

    thread_id: str = Field(default_factory=_new_ulid)
    messages: List[Message] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: ThreadMetadata = Field(default_factory=ThreadMetadata)


class ThreadManager:
    """直接使用 Redis List/Hash/String 管理 Thread。"""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[Redis] = None,
    ) -> None:
        self._redis_url = redis_url
        self._redis_client = redis_client

    async def _get_client(self) -> Redis:
        if self._redis_client is None:
            self._redis_client = get_redis_client(self._redis_url)
        return self._redis_client

    def _get_thread_keys(self, thread_id: str) -> Dict[str, str]:
        return RedisKeys.all_thread_keys(thread_id)

    async def _refresh_thread_ttl(self, thread_id: str) -> None:
        client = await self._get_client()
        for key in self._get_thread_keys(thread_id).values():
            await client.expire(key, THREAD_TTL_SECONDS)

    async def create_thread(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        metadata = ThreadMetadata(user_id=user_id, session_id=session_id, tags=list(tags or []))
        thread = Thread(context=dict(initial_context or {}), metadata=metadata)
        if not await self._save_thread_state(thread):
            raise RuntimeError("Failed to persist new thread")
        return thread.thread_id

    async def get_thread(self, thread_id: str) -> Optional[Thread]:
        try:
            client = await self._get_client()
            keys = self._get_thread_keys(thread_id)
            if not await client.exists(keys["metadata"]):
                return None
            messages_data = await client.lrange(keys["messages"], 0, -1)
            context_data = await client.hgetall(keys["context"])
            metadata_data = await client.hgetall(keys["metadata"])

            messages: List[Message] = []
            for raw_message in messages_data:
                try:
                    messages.append(Message.model_validate_json(_decode(raw_message)))
                except Exception:
                    continue

            context = {
                str(_decode(key)): _deserialize_context_value(value)
                for key, value in context_data.items()
            }
            metadata_dict = {
                str(_decode(key)): _decode(value) for key, value in metadata_data.items()
            }
            if metadata_dict.get("tags"):
                metadata_dict["tags"] = json.loads(metadata_dict["tags"])
            metadata = ThreadMetadata.model_validate(metadata_dict)
            return Thread(
                thread_id=thread_id,
                messages=messages,
                context=context,
                metadata=metadata,
            )
        except Exception:
            return None

    async def update_thread_subject(self, thread_id: str, original_message: str) -> bool:
        try:
            client = await self._get_client()
            keys = self._get_thread_keys(thread_id)
            if not await client.exists(keys["metadata"]):
                return False
            subject = str(original_message or "").strip()
            if len(subject) > 50:
                subject = subject[:50].rstrip() + "..."
            await client.hset(keys["metadata"], "subject", subject)
            await client.hset(keys["metadata"], "updated_at", _now_iso())
            await self._refresh_thread_ttl(thread_id)
            return True
        except Exception:
            return False

    async def update_thread_context(
        self,
        thread_id: str,
        context_updates: Dict[str, Any],
        merge: bool = True,
    ) -> bool:
        try:
            client = await self._get_client()
            keys = self._get_thread_keys(thread_id)
            if not await client.exists(keys["metadata"]):
                return False
            context_to_save: Dict[str, str] = {}
            if merge:
                existing = await client.hgetall(keys["context"])
                context_to_save.update(
                    {
                        str(_decode(key)): str(_decode(value))
                        for key, value in existing.items()
                    }
                )
            else:
                await client.delete(keys["context"])
            context_to_save.update(
                {
                    str(key): _serialize_context_value(value)
                    for key, value in (context_updates or {}).items()
                }
            )
            if context_to_save:
                await client.hset(keys["context"], mapping=context_to_save)
            await client.hset(keys["metadata"], "updated_at", _now_iso())
            await self._refresh_thread_ttl(thread_id)
            return True
        except Exception:
            return False

    async def append_messages(self, thread_id: str, messages: List[Dict[str, Any]]) -> bool:
        try:
            client = await self._get_client()
            keys = self._get_thread_keys(thread_id)
            if not await client.exists(keys["metadata"]):
                return False
            for raw in messages or []:
                if not isinstance(raw, dict) or not raw.get("content"):
                    continue
                role = str(raw.get("role") or "user")
                if role not in {"user", "assistant", "system"}:
                    role = "user"
                metadata = raw.get("metadata")
                message_id = raw.get("message_id") or (
                    metadata.get("message_id") if isinstance(metadata, dict) else None
                )
                message = Message(
                    message_id=message_id,
                    role=role,
                    content=str(raw["content"]),
                    metadata=metadata if isinstance(metadata, dict) else None,
                )
                await client.rpush(keys["messages"], message.model_dump_json())
            await client.hset(keys["metadata"], "updated_at", _now_iso())
            await self._refresh_thread_ttl(thread_id)
            return True
        except Exception:
            return False

    async def _save_thread_state(self, thread_state: Thread) -> bool:
        try:
            client = await self._get_client()
            keys = self._get_thread_keys(thread_state.thread_id)
            async with client.pipeline(transaction=True) as pipeline:
                pipeline.delete(keys["messages"])
                for message in thread_state.messages:
                    pipeline.rpush(keys["messages"], message.model_dump_json())
                pipeline.delete(keys["context"])
                clean_context = {
                    str(key): _serialize_context_value(value)
                    for key, value in thread_state.context.items()
                    if key != "messages"
                }
                if clean_context:
                    pipeline.hset(keys["context"], mapping=clean_context)
                metadata = thread_state.metadata.model_dump()
                metadata["tags"] = json.dumps(metadata["tags"], ensure_ascii=False)
                pipeline.hset(
                    keys["metadata"],
                    mapping={
                        key: str(value) if value is not None else ""
                        for key, value in metadata.items()
                    },
                )
                for key in keys.values():
                    pipeline.expire(key, THREAD_TTL_SECONDS)
                await pipeline.execute()
            return True
        except Exception:
            return False

    async def set_message_trace(
        self,
        message_id: str,
        tool_envelopes: List[Dict[str, Any]],
        otel_trace_id: Optional[str] = None,
    ) -> bool:
        from redis_sre_agent.agent.models import DecisionTrace

        trace = DecisionTrace(
            message_id=message_id,
            tool_envelopes=list(tool_envelopes or []),
            otel_trace_id=otel_trace_id,
            created_at=_now_iso(),
        )
        try:
            client = await self._get_client()
            await client.setex(
                RedisKeys.message_decision_trace(message_id),
                MESSAGE_TRACE_TTL_SECONDS,
                trace.model_dump_json(),
            )
            return True
        except Exception:
            return False

    async def get_message_trace(self, message_id: str) -> Optional[Dict[str, Any]]:
        try:
            client = await self._get_client()
            raw = await client.get(RedisKeys.message_decision_trace(message_id))
            return json.loads(_decode(raw)) if raw else None
        except Exception:
            return None


async def create_thread(
    *,
    query: str,
    user_id: Optional[str] = None,
    initial_context: Optional[Dict[str, Any]] = None,
    redis_client: Optional[Redis] = None,
    task_func: Optional[Any] = None,
) -> Dict[str, Any]:
    manager = ThreadManager(redis_client=redis_client)
    thread_id = await manager.create_thread(
        user_id=user_id,
        session_id=f"cli:{_new_ulid()}",
        initial_context=initial_context or {},
        tags=["cli"],
    )
    await manager.update_thread_subject(thread_id, query)
    if task_func is not None:
        await task_func(thread_id=thread_id, message=query, context=initial_context or {})
    return {"thread_id": thread_id, "message": "Thread created and queued for analysis"}


async def continue_thread(
    *,
    thread_id: str,
    query: str,
    redis_client: Optional[Redis] = None,
    task_func: Optional[Any] = None,
) -> Dict[str, Any]:
    manager = ThreadManager(redis_client=redis_client)
    thread = await manager.get_thread(thread_id)
    if thread is None:
        raise ValueError(f"Thread {thread_id} not found")
    if task_func is not None:
        await task_func(thread_id=thread_id, message=query, context=thread.context)
    return {"thread_id": thread_id, "message": "Continuation queued for processing"}
