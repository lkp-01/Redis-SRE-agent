"""External MCP Client 的本地真实 stdio 协议闭环。"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from redis_sre_agent.agent.helpers import build_result_envelope
from redis_sre_agent.agent.tool_execution import execute_tool_calls_with_gate
from redis_sre_agent.core.config import Settings
from redis_sre_agent.tools.manager import ToolManager
from redis_sre_agent.tools.mcp import provider as provider_module

REMOTE_ERROR_SENTINEL = "FAKE_MCP_REMOTE_ERROR_SENTINEL"
FAKE_SERVER_PATH = Path(__file__).with_name("fake_mcp_server.py").resolve()


def _settings_for_local_server(
    lifecycle_file: Path,
    *,
    tool_timeout: int = 10,
    include_slow_tool: bool = False,
) -> Settings:
    tools = {
        "read_status": {"capability": "diagnostics", "action_kind": "read"},
        "write_status": {"capability": "diagnostics", "action_kind": "write"},
        "large_result": {"capability": "diagnostics", "action_kind": "read"},
        "raise_secret_error": {
            "capability": "diagnostics",
            "action_kind": "read",
        },
    }
    if include_slow_tool:
        tools["slow_status"] = {
            "capability": "diagnostics",
            "action_kind": "read",
        }
    return Settings(
        _env_file=None,
        rag_enabled=False,
        tool_timeout=tool_timeout,
        mcp_servers={
            "local_stdio": {
                "command": sys.executable,
                "args": ["-u", str(FAKE_SERVER_PATH)],
                "env": {"FAKE_MCP_LIFECYCLE_FILE": str(lifecycle_file)},
                "tools": tools,
            }
        },
    )


def _read_started_pid(path: Path, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            if content.startswith("started:"):
                return int(content.split(":", 1)[1])
        time.sleep(0.02)
    raise AssertionError("fake MCP server did not publish its process id")


def _process_is_running(pid: int) -> bool:
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _wait_for_process_exit(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_is_running(pid):
            return True
        time.sleep(0.02)
    return not _process_is_running(pid)


def _cleanup_process(pid: int) -> None:
    if not _process_is_running(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


def _tool_name(names: list[str], suffix: str) -> str:
    return next(name for name in names if name.startswith("mcp_") and name.endswith(suffix))


@pytest.mark.asyncio
async def test_stdio_protocol_registers_calls_envelopes_and_closes_process(
    monkeypatch,
    tmp_path: Path,
    caplog,
    capsys,
) -> None:
    import redis_sre_agent.tools.manager as manager_module

    lifecycle_file = tmp_path / "mcp-lifecycle.txt"
    config = _settings_for_local_server(lifecycle_file)
    monkeypatch.setattr(manager_module, "settings", config)
    monkeypatch.setattr(provider_module, "settings", config)
    pid: int | None = None

    try:
        async with ToolManager(thread_id="integration-mcp-stdio") as manager:
            definitions = manager.get_tools_for_llm()
            names = [tool.name for tool in definitions]
            pid = _read_started_pid(lifecycle_file)

            read_name = _tool_name(names, "read_status")
            large_name = _tool_name(names, "large_result")
            error_name = _tool_name(names, "raise_secret_error")
            assert not any(name.endswith("write_status") for name in names)

            messages = await execute_tool_calls_with_gate(
                tool_manager=manager,
                tool_calls=[
                    {
                        "id": "stdio_read_call",
                        "name": read_name,
                        "args": {"detail": True},
                    }
                ],
            )
            envelope = build_result_envelope(
                read_name,
                {"detail": True},
                messages[0],
                {tool.name: tool for tool in definitions},
            )
            large_result = await manager.resolve_tool_call(large_name, {})
            error_result = await manager.resolve_tool_call(error_name, {})

            rendered_large = json.dumps(
                large_result.get("data", {}),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            assert envelope["status"] == "success"
            assert envelope["data"]["data"] == {
                "status": "ok",
                "detail": True,
                "source": "local-fake-mcp",
            }
            assert len(rendered_large) + len(large_result.get("text", "")) <= 32000
            assert error_result == {"status": "error", "error": "mcp_tool_error"}
            with pytest.raises(ValueError, match="Unknown tool"):
                await manager.resolve_tool_call("write_status", {"value": "blocked"})

        assert pid is not None
        exited = _wait_for_process_exit(pid)
        assert exited, "stdio MCP subprocess remained alive after ToolManager exit"
    finally:
        if pid is not None:
            _cleanup_process(pid)

    captured = capsys.readouterr()
    combined_output = f"{captured.out}\n{captured.err}\n{caplog.text}"
    assert REMOTE_ERROR_SENTINEL not in combined_output
    assert "Traceback" not in combined_output


@pytest.mark.asyncio
async def test_missing_stdio_server_isolated_from_builtin_redis_discovery(
    monkeypatch,
    tmp_path: Path,
    caplog,
) -> None:
    import redis_sre_agent.tools.manager as manager_module

    missing_path = tmp_path / "missing_fake_mcp_server.py"
    config = Settings(
        _env_file=None,
        rag_enabled=False,
        mcp_servers={
            "missing": {
                "command": sys.executable,
                "args": [str(missing_path)],
                "tools": {"read_status": {"action_kind": "read"}},
            }
        },
    )
    monkeypatch.setattr(manager_module, "settings", config)
    monkeypatch.setattr(provider_module, "settings", config)

    async with ToolManager() as manager:
        names = [tool.name for tool in manager.get_tools_for_llm()]

    assert any(name.endswith("resolve_redis_targets") for name in names)
    assert not any(name.startswith("mcp_") for name in names)
    assert "mcp_entrypoint_unavailable" in caplog.text
    assert str(missing_path) not in caplog.text
    assert "Traceback" not in caplog.text


@pytest.mark.asyncio
async def test_stdio_tool_timeout_returns_safe_code_and_process_still_closes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import redis_sre_agent.tools.manager as manager_module

    lifecycle_file = tmp_path / "mcp-timeout-lifecycle.txt"
    config = _settings_for_local_server(
        lifecycle_file,
        tool_timeout=2,
        include_slow_tool=True,
    )
    monkeypatch.setattr(manager_module, "settings", config)
    monkeypatch.setattr(provider_module, "settings", config)
    pid: int | None = None

    try:
        async with ToolManager(thread_id="integration-mcp-timeout") as manager:
            names = [tool.name for tool in manager.get_tools_for_llm()]
            slow_name = _tool_name(names, "slow_status")
            pid = _read_started_pid(lifecycle_file)
            result = await manager.resolve_tool_call(slow_name, {"delay_seconds": 10.0})

        assert result == {"status": "error", "error": "mcp_timeout"}
        assert pid is not None
        exited = _wait_for_process_exit(pid)
        assert exited, "timed-out stdio MCP subprocess remained alive after Manager exit"
    finally:
        if pid is not None:
            _cleanup_process(pid)
