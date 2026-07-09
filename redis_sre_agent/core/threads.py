"""Thread state management for CLI/Agent conversations.

本阶段先恢复 original 的 Thread 生命周期接缝：创建/读取 thread、追加消息、
保存 context 和 message trace。完整 Redis 持久化、搜索索引和标题 LLM 生成留作
后续阶段；当前实现使用轻量内存 backend，测试不会访问真实 Redis。
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Message(BaseModel):
    """Thread 中的一条可见对话消息。"""

    message_id: Optional[str] = Field(default=None)
    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)

    @model_validator(mode="after")
    def populate_message_id(self) -> "Message":
        if self.message_id is None:
            metadata_message_id = self.metadata.get("message_id")
            object.__setattr__(
                self,
                "message_id",
                str(metadata_message_id or uuid4()),
            )
        return self


class ThreadMetadata(BaseModel):
    """Thread 元数据，字段名贴近 original。"""

    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    subject: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class Thread(BaseModel):
    """一次多轮诊断会话。"""

    thread_id: str = Field(default_factory=lambda: str(uuid4()))
    messages: List[Message] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: ThreadMetadata = Field(default_factory=ThreadMetadata)


_THREADS: Dict[str, Thread] = {}
_MESSAGE_TRACES: Dict[str, Dict[str, Any]] = {}


def _clone_thread(thread: Thread) -> Thread:
    return Thread.model_validate(thread.model_dump(mode="json"))


class ThreadManager:
    """轻量 ThreadManager。

    redis_client 参数保留 original 构造形状；当前阶段不使用它，避免 CLI 测试连接真实
    Redis。后续恢复 Redis backend 时可在这些方法内部替换存储实现。
    """

    def __init__(self, redis_client: Optional[Any] = None):
        self._redis = redis_client

    async def create_thread(
        self,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        metadata = ThreadMetadata(user_id=user_id, session_id=session_id, tags=list(tags or []))
        thread = Thread(context=copy.deepcopy(initial_context or {}), metadata=metadata)
        _THREADS[thread.thread_id] = thread
        return thread.thread_id

    async def get_thread(self, thread_id: str) -> Optional[Thread]:
        thread = _THREADS.get(thread_id)
        return _clone_thread(thread) if thread is not None else None

    async def update_thread_subject(self, thread_id: str, original_message: str) -> bool:
        thread = _THREADS.get(thread_id)
        if thread is None:
            return False
        subject = str(original_message or "").strip()
        if len(subject) > 50:
            subject = subject[:50].rstrip() + "..."
        thread.metadata.subject = subject or None
        thread.metadata.updated_at = _now_iso()
        return True

    async def update_thread_context(
        self,
        thread_id: str,
        context_updates: Dict[str, Any],
        merge: bool = True,
    ) -> bool:
        thread = _THREADS.get(thread_id)
        if thread is None:
            return False
        if merge:
            thread.context.update(copy.deepcopy(context_updates or {}))
        else:
            thread.context = copy.deepcopy(context_updates or {})
        thread.metadata.updated_at = _now_iso()
        return True

    async def append_messages(self, thread_id: str, messages: List[Dict[str, Any]]) -> bool:
        thread = _THREADS.get(thread_id)
        if thread is None:
            return False
        for raw in messages or []:
            metadata = dict(raw.get("metadata") or {})
            message_id = raw.get("message_id") or metadata.get("message_id")
            msg = Message(
                message_id=message_id,
                role=str(raw.get("role") or "assistant"),
                content=str(raw.get("content") or ""),
                metadata=metadata,
            )
            thread.messages.append(msg)
        thread.metadata.updated_at = _now_iso()
        return True

    async def set_message_trace(
        self,
        *,
        message_id: str,
        tool_envelopes: List[Dict[str, Any]],
        otel_trace_id: Optional[str] = None,
    ) -> bool:
        _MESSAGE_TRACES[message_id] = {
            "message_id": message_id,
            "tool_envelopes": copy.deepcopy(tool_envelopes or []),
            "otel_trace_id": otel_trace_id,
            "created_at": _now_iso(),
        }
        return True

    async def get_message_trace(self, message_id: str) -> Optional[Dict[str, Any]]:
        trace = _MESSAGE_TRACES.get(message_id)
        return copy.deepcopy(trace) if trace is not None else None


async def create_thread(
    *,
    query: str,
    user_id: Optional[str] = None,
    initial_context: Optional[Dict[str, Any]] = None,
    redis_client: Optional[Any] = None,
    task_func: Optional[Any] = None,
) -> Dict[str, Any]:
    """original 顶层 create_thread helper 的轻量插槽。"""

    manager = ThreadManager(redis_client=redis_client)
    thread_id = await manager.create_thread(
        user_id=user_id,
        session_id=f"cli:{uuid4()}",
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
    redis_client: Optional[Any] = None,
    task_func: Optional[Any] = None,
) -> Dict[str, Any]:
    """original continue_thread helper 的轻量插槽。"""

    manager = ThreadManager(redis_client=redis_client)
    thread = await manager.get_thread(thread_id)
    if thread is None:
        raise ValueError(f"Thread {thread_id} not found")
    if task_func is not None:
        await task_func(thread_id=thread_id, message=query, context=thread.context)
    return {"thread_id": thread_id, "message": "Continuation queued for processing"}
