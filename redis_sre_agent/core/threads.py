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
            # 1. 获取异步 Redis 客户端实例
            client = await self._get_client()

            # 2. 根据 thread_id 生成该会话在 Redis 中对应的所有 Key（包括 metadata, context, messages 等的键名）
            keys = self._get_thread_keys(thread_id)

            # 3. 检查 Redis 中是否存在该 Thread 的元数据 Key。如果不存在，说明该 thread_id 无效或已过期，直接返回 False
            if not await client.exists(keys["metadata"]):
                return False

            # 4. 初始化一个字典，用于存放最终需要写入 Redis Hash 的键值对（Redis Hash 的 field 和 value 都必须是字符串）
            context_to_save: Dict[str, str] = {}

            # 5. 如果 merge 为 True（默认值），执行“合并/追加”逻辑
            if merge:
                # 从 Redis Hash 中取出该 Thread 当前已有的所有上下文数据
                existing = await client.hgetall(keys["context"])

                # 将原有数据解码为字符串，并预先放入待保存的字典中，确保老数据不丢失
                context_to_save.update(
                    {
                        str(_decode(key)): str(_decode(value))
                        for key, value in existing.items()
                    }
                )
            # 6. 如果 merge 为 False，执行“完全覆盖”逻辑
            else:
                # 直接从 Redis 中删除该 Thread 原有的整个 context Hash 键，清空老数据
                await client.delete(keys["context"])

            # 7. 将用户传入的、需要更新的 context_updates 字典进行序列化，并合并/覆盖到待保存的字典中
            context_to_save.update(
                {
                    # 确保 Key 是字符串，Value 通过 _serialize_context_value 转换（如果是 dict/list 会转为 JSON 字符串）
                    str(key): _serialize_context_value(value)
                    for key, value in (context_updates or {}).items()
                }
            )

            # 8. 如果最终有需要保存的数据，则调用 Redis 的 hset 命令将最新的映射关系一次性写入 Hash 中
            if context_to_save:
                await client.hset(keys["context"], mapping=context_to_save)

            # 9. 更新该 Thread 元数据中的“最后更新时间（updated_at）”为当前最新的 UTC 时间
            await client.hset(keys["metadata"], "updated_at", _now_iso())

            # 10. 重新为该 Thread 相关的所有 Redis Key 刷新过期时间（TTL），防止会话因为长期不活跃而被 Redis 自动删除
            await self._refresh_thread_ttl(thread_id)

            # 11. 整个更新流程成功完成，返回 True
            return True

        # 12. 捕获期间发生的所有异常（如 Redis 连接断开、序列化失败等），防止程序崩溃，并返回 False 告知调用方更新失败
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
        # 从 redis_sre_agent.agent.models 模块中延迟导入 DecisionTrace 追踪模型
        # 延迟导入通常是为了避免循环依赖（Circular Import）
        from redis_sre_agent.agent.models import DecisionTrace

        # 实例化一个决策追踪对象（DecisionTrace）
        # 1. message_id: 绑定当前追踪记录属于哪一条消息
        # 2. tool_envelopes: 记录当前轮次中 Agent 调用工具的详细信封/数据包列表
        # 3. otel_trace_id: OpenTelemetry 的分布式链路追踪 ID（若有），用于和外部微服务链路串联
        # 4. created_at: 生成当前 UTC 时间的 ISO 字符串，记录追踪创建时间
        trace = DecisionTrace(
            message_id=message_id,
            tool_envelopes=list(tool_envelopes or []),
            otel_trace_id=otel_trace_id,
            created_at=_now_iso(),
        )
        try:
            # 获取异步 Redis 客户端实例
            client = await self._get_client()

            # 将 trace 对象序列化为 JSON 字符串，并通过 setex 命令写入 Redis
            # 1. RedisKeys.message_decision_trace(message_id): 生成该消息追踪专属的 Redis Key
            # 2. MESSAGE_TRACE_TTL_SECONDS: 过期时间（代码上方定义为 7 天），过期后 Redis 自动清理
            # 3. trace.model_dump_json(): 将 Pydantic 模型转换为 JSON 文本进行持久化
            await client.setex(
                RedisKeys.message_decision_trace(message_id),
                MESSAGE_TRACE_TTL_SECONDS,
                trace.model_dump_json(),
            )
            # 写入成功，返回 True
            return True
        except Exception:
            # 捕获期间发生的所有异常（如网络抖动、Redis 宕机等），防止阻塞主业务流程，并返回 False
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
