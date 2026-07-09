"""Stage 5 Task 接口形状测试。"""

from __future__ import annotations

import pytest

from redis_sre_agent.core.tasks import (
    TaskManager,
    TaskStatus,
    delete_task,
    get_task_by_id,
    list_tasks,
)


@pytest.mark.asyncio
async def test_task_manager_keeps_original_style_task_interfaces() -> None:
    manager = TaskManager()
    task_id = await manager.create_task(
        thread_id="thread-stage5-task",
        user_id="user-stage5-task",
        subject="check redis",
    )
    await manager.add_task_update(task_id, "started", metadata={"message_id": "msg-stage5"})
    await manager.set_task_result(task_id, {"message_id": "msg-stage5", "response": "ok"})

    detail = await get_task_by_id(task_id=task_id)
    listed = await list_tasks(user_id="user-stage5-task")
    deleted = await delete_task(task_id=task_id)

    assert detail["task_id"] == task_id
    assert detail["thread_id"] == "thread-stage5-task"
    assert detail["status"] == TaskStatus.DONE
    assert detail["metadata"]["user_id"] == "user-stage5-task"
    assert any(item["task_id"] == task_id for item in listed)
    assert deleted == {"task_id": task_id, "deleted": True}
