"""CLI logging helpers.

original 在 CLI 顶层统一配置日志并避免重复记录异常。当前裁剪版保留这些函数名，
实现保持轻量，不引入 observability 后续模块。
"""

from __future__ import annotations

import logging
from typing import Any

#全局状态机制 _LOGGED_EXCEPTION_IDS，可以避免嵌套导致的重复打印
_LOGGED_EXCEPTION_IDS: set[int] = set()

#
def configure_cli_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def log_cli_exception(logger_name: str, message: str, exc: BaseException) -> None:
    _LOGGED_EXCEPTION_IDS.add(id(exc)) #id(obj) 函数会返回该对象在内存中的唯一整数 ID。
    logging.getLogger(logger_name).exception(message)


def was_cli_exception_logged(exc: BaseException) -> bool:
    return id(exc) in _LOGGED_EXCEPTION_IDS
