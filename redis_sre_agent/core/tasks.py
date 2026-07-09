"""Task state slot for per-turn agent work.

original 中 Task 表示 Thread 下的一次异步 Agent turn。当前 Stage 5 CLI 直接同步运行
Agent，不启动 worker；这里保留 TaskManager 接口形状，供后续后台任务阶段接回。
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from redis_sre_agent.core.threads import ThreadManager


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskMetadata(BaseModel):
    created_at: str = Field(default_factory=_now_iso)
    updated_at: Optional[str] = None
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    subject: Optional[str] = None


class TaskUpdate(BaseModel):
    timestamp: str = Field(default_factory=_now_iso)
    message: str
    update_type: str = "progress"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskState(BaseModel):
    task_id: str
    thread_id: str
    status: TaskStatus = TaskStatus.QUEUED
    updates: List[TaskUpdate] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: TaskMetadata = Field(default_factory=TaskMetadata)


_TASKS: Dict[str, TaskState] = {}


def _clone_task(task: TaskState) -> TaskState:
    return TaskState.model_validate(task.model_dump(mode="json"))


class TaskManager:
    """轻量任务管理器，保留 original 方法名。"""

    def __init__(self, redis_client: Optional[Any] = None):
        self._redis = redis_client

    async def create_task(
        self,
        *,
        thread_id: str,
        user_id: Optional[str] = None,
        subject: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        task_id = str(uuid4())
        task_metadata = TaskMetadata(thread_id=thread_id, user_id=user_id, subject=subject)
        if metadata:
            for key, value in metadata.items():
                if hasattr(task_metadata, key):
                    setattr(task_metadata, key, value)
        _TASKS[task_id] = TaskState(
            task_id=task_id,
            thread_id=thread_id,
            metadata=task_metadata,
        )
        return task_id

    async def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        task = _TASKS.get(task_id)
        if task is None:
            return False
        task.status = status
        task.metadata.updated_at = _now_iso()
        return True

    async def add_task_update(
        self,
        task_id: str,
        message: str,
        update_type: str = "progress",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        task = _TASKS.get(task_id)
        if task is None:
            return False
        task.updates.append(
            TaskUpdate(message=message, update_type=update_type, metadata=metadata or {})
        )
        task.metadata.updated_at = _now_iso()
        return True

    async def set_task_result(self, task_id: str, result: Dict[str, Any]) -> bool:
        task = _TASKS.get(task_id)
        if task is None:
            return False
        task.result = copy.deepcopy(result)
        task.status = TaskStatus.DONE
        task.metadata.updated_at = _now_iso()
        return True

    async def set_task_error(self, task_id: str, error_message: str) -> bool:
        task = _TASKS.get(task_id)
        if task is None:
            return False
        task.error_message = error_message
        task.status = TaskStatus.FAILED
        task.metadata.updated_at = _now_iso()
        return True

    async def get_task_state(self, task_id: str) -> Optional[TaskState]:
        task = _TASKS.get(task_id)
        return _clone_task(task) if task is not None else None

    async def get_task_tool_calls(self, task: TaskState) -> Optional[List[Dict[str, Any]]]:
        if task.status != TaskStatus.DONE:
            return None
        message_id = None
        if isinstance(task.result, dict):
            message_id = task.result.get("message_id")
        if not message_id:
            return None
        trace = await ThreadManager(redis_client=self._redis).get_message_trace(str(message_id))
        if trace and isinstance(trace.get("tool_envelopes"), list):
            return trace["tool_envelopes"]
        return None


async def create_task(
    *,
    message: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    redis_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """original create_task helper 的同步阶段插槽。"""

    thread_manager = ThreadManager(redis_client=redis_client)
    created_new_thread = False
    if not thread_id:
        thread_id = await thread_manager.create_thread(
            user_id=user_id,
            session_id=f"task:{uuid4()}",
            initial_context=context or {},
            tags=["task"],
        )
        await thread_manager.update_thread_subject(thread_id, message)
        created_new_thread = True

    task_manager = TaskManager(redis_client=redis_client)
    task_id = await task_manager.create_task(thread_id=thread_id, user_id=user_id, subject=message)
    return {
        "task_id": task_id,
        "thread_id": thread_id,
        "status": TaskStatus.QUEUED,
        "message": "Thread created; task queued" if created_new_thread else "Task created and queued",
    }


async def get_task_by_id(*, task_id: str, redis_client: Optional[Any] = None) -> Dict[str, Any]:
    """original `get_task_by_id` 的轻量同步阶段入口。"""

    task = await TaskManager(redis_client=redis_client).get_task_state(task_id)
    if task is None:
        raise ValueError(f"Task {task_id} not found")
    tool_calls = await TaskManager(redis_client=redis_client).get_task_tool_calls(task)
    return {
        "task_id": task.task_id,
        "thread_id": task.thread_id,
        "status": task.status,
        "updates": [update.model_dump(mode="json") for update in task.updates],
        "result": copy.deepcopy(task.result),
        "error_message": task.error_message,
        "metadata": task.metadata.model_dump(mode="json"),
        "tool_calls": tool_calls or [],
    }


async def list_tasks(
    *,
    user_id: Optional[str] = None,
    status_filter: Optional[TaskStatus] = None,
    limit: int = 20,
    offset: int = 0,
    redis_client: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """列出内存 Task 记录，保留 original 的查询入口形状。"""

    del redis_client
    bounded_limit = max(1, min(int(limit), 100))
    bounded_offset = max(0, int(offset))
    tasks = sorted(
        _TASKS.values(),
        key=lambda task: task.metadata.created_at,
        reverse=True,
    )
    if user_id:
        tasks = [task for task in tasks if task.metadata.user_id in {None, "", user_id}]
    if status_filter:
        tasks = [task for task in tasks if task.status == status_filter]
    page = tasks[bounded_offset : bounded_offset + bounded_limit]
    return [
        {
            "task_id": task.task_id,
            "thread_id": task.thread_id,
            "status": task.status,
            "metadata": task.metadata.model_dump(mode="json"),
        }
        for task in page
    ]


async def delete_task(*, task_id: str, redis_client: Optional[Any] = None) -> Dict[str, Any]:
    """删除内存 Task 记录；后台队列清理留作后续阶段。"""

    del redis_client
    existed = _TASKS.pop(task_id, None) is not None
    return {"task_id": task_id, "deleted": existed}
