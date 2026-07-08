"""工具 provider 使用的轻量装饰器。"""

from __future__ import annotations

from typing import Callable


def status_update(template: str) -> Callable:
    """给 provider 方法挂一段状态提示模板。

    ToolManager 不直接理解每个工具的业务含义，所以沿用原项目做法：工具方法可以把
    一段模板挂在函数对象上，运行时再用调用参数格式化。
    """

    def decorator(func: Callable) -> Callable:
        setattr(func, "_status_update_template", template)
        return func

    return decorator
