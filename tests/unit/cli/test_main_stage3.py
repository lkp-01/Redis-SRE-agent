"""Stage 5 CLI query 入口测试。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from click.testing import CliRunner
from pydantic import SecretStr

from redis_sre_agent.cli.main import main
from redis_sre_agent.core import targets as targets_module
from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.core.threads import ThreadManager
from redis_sre_agent.tools.diagnostics.redis_command.provider import RedisCommandToolProvider


_PASSWORD = "stage5-cli-password"
_URL = f"redis://default:{_PASSWORD}@cache.internal:6379/0"


class FakeCliRedisClient:
    async def info(self, section: str | None = None) -> dict[str, Any]:
        if section == "memory":
            return {"used_memory_human": "1K", "used_memory": 1024}
        if section == "stats":
            return {"instantaneous_ops_per_sec": 3}
        if section == "replication":
            return {"role": "master"}
        return {"redis_version": "7.2.0"}

    async def client_list(self, _type: str | None = None) -> list[dict[str, Any]]:
        return [{"id": "1"}]

    async def slowlog_get(self, count: int) -> list[dict[str, Any]]:
        return []

    async def memory_stats(self) -> dict[str, Any]:
        return {"peak.allocated": 2048}

    async def config_get(self, pattern: str) -> dict[str, str]:
        return {"maxmemory": "2048", "requirepass": _PASSWORD}

    async def execute_command(self, command: str, *args: Any) -> Any:
        if command == "ROLE":
            return ["master", 0, []]
        raise RuntimeError(f"unsupported command: {command}")

    async def aclose(self) -> None:
        return None


def make_instance() -> RedisInstance:
    return RedisInstance(
        id="inst-stage5-cli",
        name="Stage 5 CLI Redis",
        connection_url=SecretStr(_URL),
        environment="test",
        usage="cache",
        description="Stage 5 fake CLI target",
    )


def test_cli_status_reports_stage_five() -> None:
    result = CliRunner().invoke(main, ["status"])

    assert result.exit_code == 0
    assert "Stage 5" in result.output


def test_cli_lazy_group_exposes_query_help_and_version() -> None:
    runner = CliRunner()

    root_help = runner.invoke(main, ["--help"])
    query_help = runner.invoke(main, ["query", "--help"])
    version = runner.invoke(main, ["version"])

    assert root_help.exit_code == 0
    assert "query" in root_help.output
    assert query_help.exit_code == 0
    assert "--instance-id" in query_help.output
    assert version.exit_code == 0
    assert "redis-sre-agent" in version.output


def test_cli_query_prints_agent_response(monkeypatch) -> None:
    async def fake_get_instances():
        return []

    async def fake_get_clusters():
        return []

    monkeypatch.setattr(targets_module, "get_instances", fake_get_instances)
    monkeypatch.setattr(targets_module, "get_clusters", fake_get_clusters)

    result = CliRunner().invoke(
        main,
        ["query", "check redis", "--target", "prod-cache", "--user-id", "u1"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "Redis 诊断摘要" in payload["response"]
    assert payload["search_results"] == []
    assert payload["tool_envelopes"][0]["name"] == "resolve_redis_targets"
    assert payload["tool_envelopes"][0]["data"]["status"] == "no_match"
    assert payload["thread_id"]

    thread = asyncio.run(ThreadManager().get_thread(payload["thread_id"]))
    assert thread is not None
    assert [message.role for message in thread.messages] == ["user", "assistant"]
    assert thread.messages[0].content == "check redis"
    assert "message_id" in thread.messages[1].metadata


def test_cli_query_can_continue_thread_history(monkeypatch) -> None:
    async def fake_get_instances():
        return []

    async def fake_get_clusters():
        return []

    monkeypatch.setattr(targets_module, "get_instances", fake_get_instances)
    monkeypatch.setattr(targets_module, "get_clusters", fake_get_clusters)

    runner = CliRunner()
    first = runner.invoke(main, ["query", "check redis", "--target", "prod-cache"])
    assert first.exit_code == 0, first.output
    thread_id = json.loads(first.output)["thread_id"]

    second = runner.invoke(
        main,
        ["query", "continue checking", "--thread-id", thread_id, "--agent", "chat"],
    )
    assert second.exit_code == 0, second.output

    thread = asyncio.run(ThreadManager().get_thread(thread_id))
    assert thread is not None
    assert [message.role for message in thread.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert thread.messages[2].content == "continue checking"


def test_cli_query_uses_agent_for_instance_scope(monkeypatch) -> None:
    instance = make_instance()

    async def fake_get_instance_by_id(instance_id: str):
        return instance if instance_id == instance.id else None

    def fake_get_client(self):
        if self._client is None:
            self._client = FakeCliRedisClient()
        return self._client

    monkeypatch.setattr("redis_sre_agent.core.instances.get_instance_by_id", fake_get_instance_by_id)
    monkeypatch.setattr(RedisCommandToolProvider, "get_client", fake_get_client)

    result = CliRunner().invoke(
        main,
        [
            "query",
            "check redis",
            "--instance-id",
            instance.id,
            "--agent",
            "chat",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    names = [envelope["name"] for envelope in payload["tool_envelopes"]]
    assert "info" in names
    assert "memory_stats" in names
    assert "client_list" in names
    assert "slowlog" in names
    assert _PASSWORD not in json.dumps(payload, ensure_ascii=False)
    assert _URL not in json.dumps(payload, ensure_ascii=False)
