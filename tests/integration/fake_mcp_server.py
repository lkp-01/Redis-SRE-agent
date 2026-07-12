"""只供测试使用的本地 stdio MCP Server，不读取项目配置或访问网络。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

REMOTE_ERROR_SENTINEL = "FAKE_MCP_REMOTE_ERROR_SENTINEL"

server = FastMCP("redis-sre-agent-local-test", log_level="CRITICAL")


@server.tool(
    annotations=ToolAnnotations(readOnlyHint=True),
    structured_output=True,
)
def read_status(detail: bool = False) -> dict[str, object]:
    """Read deterministic status from the local fake server."""

    return {"status": "ok", "detail": detail, "source": "local-fake-mcp"}


@server.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True),
    structured_output=True,
)
def write_status(value: str) -> dict[str, object]:
    """Pretend to mutate status; the client must never expose this tool."""

    return {"status": "unexpected", "value": value}


@server.tool(
    annotations=ToolAnnotations(readOnlyHint=True),
    structured_output=True,
)
def large_result() -> dict[str, object]:
    """Return content larger than the client result boundary."""

    return {"status": "ok", "payload": "x" * 40000}


@server.tool(
    annotations=ToolAnnotations(readOnlyHint=True),
    structured_output=True,
)
def raise_secret_error() -> dict[str, object]:
    """Raise an error containing a test-only sentinel for redaction checks."""

    raise RuntimeError(f"remote failure: {REMOTE_ERROR_SENTINEL}")


@server.tool(
    annotations=ToolAnnotations(readOnlyHint=True),
    structured_output=True,
)
async def slow_status(delay_seconds: float = 10.0) -> dict[str, object]:
    """Wait long enough for the client-side timeout test."""

    await asyncio.sleep(delay_seconds)
    return {"status": "ok"}


def _lifecycle_file() -> Path | None:
    raw_path = os.environ.get("FAKE_MCP_LIFECYCLE_FILE")
    return Path(raw_path) if raw_path else None


if __name__ == "__main__":
    lifecycle_file = _lifecycle_file()
    if lifecycle_file is not None:
        lifecycle_file.write_text(f"started:{os.getpid()}", encoding="utf-8")
    try:
        server.run(transport="stdio")
    finally:
        if lifecycle_file is not None:
            lifecycle_file.write_text(f"closed:{os.getpid()}", encoding="utf-8")
