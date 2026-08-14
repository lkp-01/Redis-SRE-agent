"""为离线评测提供最小、按上下文隔离的工具替身入口。"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class EvalToolDispatchResult:
    """包装已由本地 fixture 接管的一次工具调用结果。"""

    result: Any


@runtime_checkable
class EvalToolRuntime(Protocol):
    """离线评测工具替身需要实现的最小接口。"""

    async def dispatch_tool_call(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        tool_by_name: Mapping[str, Any],
        routing_table: Mapping[str, Any],
    ) -> EvalToolDispatchResult: ...


_tool_runtime: ContextVar[EvalToolRuntime | None] = ContextVar(
    "eval_tool_runtime", default=None
)


def is_eval_runtime_active() -> bool:
    """返回当前异步上下文是否在离线评测中。"""

    return _tool_runtime.get() is not None


def get_eval_provider_families() -> frozenset[str]:
    """返回当前 fixture scenario 明确声明的 provider family。"""

    runtime = _tool_runtime.get()
    if runtime is None:
        return frozenset()
    families = getattr(runtime, "provider_families", ())
    return frozenset(str(item) for item in families if str(item).strip())


@contextmanager
def eval_tool_runtime_scope(runtime: EvalToolRuntime) -> Iterator[EvalToolRuntime]:
    """仅在当前上下文安装 fixture runtime，退出后恢复原值。"""

    token = _tool_runtime.set(runtime)
    try:
        yield runtime
    finally:
        _tool_runtime.reset(token)


async def dispatch_tool_runtime_override(
    *,
    tool_name: str,
    args: dict[str, Any],
    tool_by_name: Mapping[str, Any],
    routing_table: Mapping[str, Any],
) -> EvalToolDispatchResult | None:
    """把离线评测中的工具调用交给 fixture；未声明调用必须失败。"""

    runtime = _tool_runtime.get()
    if runtime is None:
        return None
    result = runtime.dispatch_tool_call(
        tool_name=tool_name,
        args=args,
        tool_by_name=tool_by_name,
        routing_table=routing_table,
    )
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, EvalToolDispatchResult):
        raise TypeError("评测工具替身必须返回 EvalToolDispatchResult")
    return result


__all__ = [
    "EvalToolDispatchResult",
    "EvalToolRuntime",
    "dispatch_tool_runtime_override",
    "eval_tool_runtime_scope",
    "get_eval_provider_families",
    "is_eval_runtime_active",
]
