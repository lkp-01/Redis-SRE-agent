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

    message_id: Optional[str] = Field(default=None) #每个消息的标识符
    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict) #与该消息相关的额外结构化数据（例如调用工具的记录、特定的UI标记等）
    created_at: str = Field(default_factory=_now_iso)

    @model_validator(mode="after") # Pydantic 验证器：指定在模型数据组装/初始化之后 (mode="after") 自动执行。
    def populate_message_id(self) -> "Message":
        if self.message_id is None:# 检查实例在初始化时是否缺失 message_id
            metadata_message_id = self.metadata.get("message_id")# 尝试从 metadata 字典中查找是否已经自带了 message_id
            object.__setattr__( # 如果字典里有就用，没有就生成新的
                self,
                "message_id",
                str(metadata_message_id or uuid4()),
            )
        return self


class ThreadMetadata(BaseModel):
    """Thread 元数据，字段名贴近 original。包括时间、用户标识、会话表示、主题、分类列表"""

    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    subject: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class Thread(BaseModel):
    """一次多轮诊断会话。"""

    thread_id: str = Field(default_factory=lambda: str(uuid4()))# 当前对话线程的全局唯一 ID，默认使用 lambda 延迟生成一个全新的 UUID。
    messages: List[Message] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)# 存储对话级别的全局上下文状态。
    metadata: ThreadMetadata = Field(default_factory=ThreadMetadata)# 嵌套之前定义的 ThreadMetadata 模型，存储这个 Thread 的周边描述信息。

#两个全局字典
_THREADS: Dict[str, Thread] = {} #存放所有Thread实例
_MESSAGE_TRACES: Dict[str, Dict[str, Any]] = {} #存储消息级别的底层执行轨迹（比如该消息调用了什么工具等）。键为 message_id。


def _clone_thread(thread: Thread) -> Thread:
    return Thread.model_validate(thread.model_dump(mode="json"))
    # 深拷贝 Thread 对象。通过先将其导出为 JSON 字典，再让 Pydantic 重新验证生成新对象。
    # 这样做的目的是隔离外部操作，防止按引用传递导致外部的修改意外污染了 _THREADS 内存库里的原始数据。


class ThreadManager:
    """轻量 ThreadManager。

    redis_client 参数保留 original 构造形状；当前阶段不使用它，避免 CLI 测试连接真实
    Redis。后续恢复 Redis backend 时可在这些方法内部替换存储实现。
    """

    def __init__(self, redis_client: Optional[Any] = None):
        self._redis = redis_client

    async def create_thread( # 实例化一个ThreadMetadata，然后返回其id
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

    # 从内存字典中尝试获取对应的 Thread 对象。找到了就深度拷贝安全返回
    async def get_thread(self, thread_id: str) -> Optional[Thread]:
        thread = _THREADS.get(thread_id)
        return _clone_thread(thread) if thread is not None else None

    # 处理文本，更新对话的主题/标题
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

    # 更新对话的上下文
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
        # 往对话历史里追加新的对话消息（比如用户说了一句话，或者 AI 回复了一句话）
        thread = _THREADS.get(thread_id)
        if thread is None:
            return False
        for raw in messages or []:# 外部传进来的通常是原始的字典列表，我们需要遍历它们
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
    ) -> bool: # 某条消息背后，大模型背后可能调用了工具，因此专门设置了全局map存消息背后的轨迹
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
    redis_client: Optional[Any] = None,# 可选参数：未来要接入的 Redis 客户端实例
    task_func: Optional[Any] = None, # 可选参数：触发后续业务逻辑的后台任务函数（核心解耦设计）
) -> Dict[str, Any]:
    """original 顶层 create_thread helper 的轻量插槽。"""

    manager = ThreadManager(redis_client=redis_client)
    thread_id = await manager.create_thread(
        user_id=user_id,
        session_id=f"cli:{uuid4()}",
        initial_context=initial_context or {},
        tags=["cli"],
    )
    await manager.update_thread_subject(thread_id, query) #把第一句话变成主题
    if task_func is not None:  # 扔进去问题和初始上下文
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
    if task_func is not None:# 把新问题和旧窗口的记忆扔进去实现上下文回答
        await task_func(thread_id=thread_id, message=query, context=thread.context)
    return {"thread_id": thread_id, "message": "Continuation queued for processing"}
