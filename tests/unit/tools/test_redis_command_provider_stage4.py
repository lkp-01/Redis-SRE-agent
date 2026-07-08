"""Stage 4 Redis command provider 测试。

这些测试只使用内存 fake client，目的是验证 provider 能把 Redis 只读命令结果整理成
结构化 evidence，同时保证错误、配置和慢日志输出不会带出敏感值。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from pydantic import SecretStr

from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.tools.diagnostics.redis_command import provider as provider_module
from redis_sre_agent.tools.diagnostics.redis_command.provider import RedisCommandToolProvider


_PASSWORD = "stage4-provider-password"
_CONFIG_SECRET = "stage4-config-secret"
_SLOWLOG_SECRET = "stage4-slowlog-secret"
_TOKEN_VALUE = "stage4-token-value"
_URL = f"redis://default:{_PASSWORD}@cache.internal:6379/0"


def _assert_no_sensitive_markers(value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    for label, marker in {
        "redis_password": _PASSWORD,
        "config_secret": _CONFIG_SECRET,
        "slowlog_secret": _SLOWLOG_SECRET,
        "token": _TOKEN_VALUE,
        "connection_url": _URL,
    }.items():
        if marker in payload:
            pytest.fail(f"sensitive marker leaked: {label}")


class FakePipeline:
    def __init__(self, client: "FakeRedisClient") -> None:
        self.client = client
        self.commands: list[tuple[str, Any]] = []

    def randomkey(self) -> None:
        self.commands.append(("randomkey", None))

    def type(self, key: str) -> None:
        self.commands.append(("type", key))

    async def execute(self) -> list[Any]:
        if not self.commands:
            return []
        operation = self.commands[0][0]
        if operation == "randomkey":
            return [
                self.client.random_keys[index % len(self.client.random_keys)]
                for index, _ in enumerate(self.commands)
            ]
        if operation == "type":
            return [self.client.key_types.get(str(key), "string") for _, key in self.commands]
        return []


class FakeRedisClient:
    def __init__(
        self,
        *,
        info_error: Exception | None = None,
        memory_stats_error: Exception | None = None,
    ) -> None:
        self.info_error = info_error
        self.memory_stats_error = memory_stats_error
        self.closed = False
        self.random_keys = ["session:1", "cart:1", "session:1", "stream:1", None]
        self.key_types = {
            "session:1": "string",
            "cart:1": "hash",
            "stream:1": "stream",
        }

    async def info(self, section: str | None = None) -> dict[str, Any]:
        if self.info_error is not None:
            raise self.info_error
        if section == "memory":
            return {"used_memory": 1024, "maxmemory_policy": "allkeys-lru"}
        if section == "replication":
            return {
                "role": "master",
                "connected_slaves": 1,
                "slave0": "ip=10.0.0.2,port=6380,state=online",
            }
        return {"redis_version": "7.2.0", "role": "master"}

    async def slowlog_get(self, count: int) -> list[dict[str, Any]]:
        entries = [
            {
                "id": 10,
                "start_time": 1_700_000_000,
                "duration": 1250,
                "command": [b"AUTH", _SLOWLOG_SECRET.encode()],
                "client_address": "10.0.0.10:5000",
                "client_name": "worker-a",
            },
            {
                "id": 11,
                "start_time": 1_700_000_001,
                "duration": 900,
                "command": ["SET", "api_token", _TOKEN_VALUE],
                "client_address": "10.0.0.11:5000",
                "client_name": "worker-b",
            },
        ]
        return entries[:count]

    async def acl_log(self, count: int) -> list[dict[str, Any]]:
        entries = [
            {
                "count": 2,
                "reason": "auth",
                "context": "toplevel",
                "object": "AUTH",
                "username": "default",
                "age-seconds": 7,
                "client-info": "addr=10.0.0.10:5000 name=worker-a",
            },
            {
                "count": 1,
                "reason": "command",
                "context": "multi",
                "object": "CONFIG",
                "username": "readonly",
                "age-seconds": 3,
                "client-info": "addr=10.0.0.11:5000 name=worker-b",
            },
        ]
        return entries[:count]

    async def config_get(self, pattern: str) -> dict[str, str]:
        return {
            "maxmemory": "1048576",
            "requirepass": _CONFIG_SECRET,
            "masterauth": _CONFIG_SECRET,
            "api_token": _TOKEN_VALUE,
            "normal_setting": "plain-value",
        }

    async def client_list(self, _type: str | None = None) -> list[dict[str, Any]]:
        clients = [
            {
                "id": "42",
                "addr": "10.0.0.10:5000",
                "laddr": "127.0.0.1:6379",
                "name": "worker-a",
                "cmd": "get",
            },
            {
                "id": "43",
                "addr": "10.0.0.11:5000",
                "laddr": "127.0.0.1:6379",
                "name": "worker-b",
                "cmd": "set",
            },
        ]
        return clients if _type is None else clients[:1]

    async def cluster(self, subcommand: str) -> Any:
        if subcommand == "INFO":
            return {"cluster_state": "ok", "cluster_slots_assigned": 16384}
        if subcommand == "NODES":
            return "node-1 10.0.0.1:6379@16379 master - 0 0 1 connected\n"
        raise RuntimeError(f"unsupported cluster command: {subcommand}")

    async def execute_command(self, command: str, *args: Any) -> Any:
        if command == "ROLE":
            return ["master", 0, []]
        if command == "FT._LIST":
            return [b"idx:users", "idx:orders"]
        if command == "FT.INFO":
            return [
                b"index_name",
                args[0],
                b"num_docs",
                "2",
                b"fields",
                [b"title", b"TEXT"],
            ]
        raise RuntimeError(f"unsupported command: {command}")

    async def memory_stats(self) -> dict[str, Any]:
        if self.memory_stats_error is not None:
            raise self.memory_stats_error
        return {"peak.allocated": 2048, "clients.normal": 64}

    def pipeline(self, transaction: bool = False) -> FakePipeline:
        return FakePipeline(self)

    async def client_id(self) -> int:
        return 42

    async def aclose(self) -> None:
        self.closed = True


def make_instance() -> RedisInstance:
    return RedisInstance(
        id="inst-stage4",
        name="Stage 4 Redis",
        connection_url=SecretStr(_URL),
        environment="test",
        usage="diagnostics",
        description="Stage 4 fake Redis target",
    )


def make_provider(fake_client: FakeRedisClient | None = None) -> RedisCommandToolProvider:
    provider = RedisCommandToolProvider(redis_instance=make_instance())
    provider._client = fake_client or FakeRedisClient()
    return provider


def test_create_tool_schemas_exposes_stage4_tools() -> None:
    provider = make_provider()
    suffixes = {tool.name.rsplit("_", 1)[-1] for tool in provider.create_tool_schemas()}
    names = {tool.name for tool in provider.create_tool_schemas()}

    assert any(name.endswith("info") for name in names)
    assert "slowlog" in suffixes
    assert any(name.endswith("acl_log") for name in names)
    assert any(name.endswith("config_get") for name in names)
    assert any(name.endswith("client_list") for name in names)
    assert any(name.endswith("cluster_info") for name in names)
    assert any(name.endswith("replication_info") for name in names)
    assert any(name.endswith("memory_stats") for name in names)
    assert any(name.endswith("sample_keys") for name in names)
    assert any(name.endswith("search_indexes") for name in names)
    assert any(name.endswith("search_index_info") for name in names)


@pytest.mark.asyncio
async def test_stage4_tools_return_structured_evidence() -> None:
    provider = make_provider()

    info = await provider.info(section="memory")
    slowlog = await provider.slowlog(count=2)
    acl_log = await provider.acl_log(count=2)
    config = await provider.config_get("maxmemory*")
    clients = await provider.client_list()
    cluster = await provider.cluster_info()
    replication = await provider.replication_info()
    memory = await provider.memory_stats()
    sample = await provider.sample_keys(count=3)
    indexes = await provider.search_indexes()
    index_info = await provider.search_index_info("idx:users")

    assert info["status"] == "success"
    assert info["section"] == "memory"
    assert info["data"]["used_memory"] == 1024
    assert slowlog["count"] == 2
    assert slowlog["entries"][0]["duration_us"] == 1250
    assert isinstance(slowlog["entries"][0]["command"], str)
    assert acl_log["count"] == 2
    assert acl_log["entries"][0]["reason"] == "auth"
    assert config["pattern"] == "maxmemory*"
    assert config["config"]["maxmemory"] == "1048576"
    assert config["count"] == 5
    assert clients["client_type"] == "all"
    assert clients["count"] == 2
    assert cluster["cluster_info"]["cluster_state"] == "ok"
    assert replication["info"]["role"] == "master"
    assert replication["role"]["type"] == "master"
    assert memory["stats"]["peak.allocated"] == 2048
    assert memory["interpretation_notes"]
    assert memory["canonical_sources"]["client_counts"] == ["INFO clients", "CLIENT LIST"]
    assert sample["sampled_count"] == 3
    assert sample["keys"]
    assert sample["type_distribution"] == {"string": 1, "hash": 1, "stream": 1}
    assert sample["limit_applied"] is False
    assert indexes["indexes"] == ["idx:users", "idx:orders"]
    assert index_info["info"]["index_name"] == "idx:users"
    assert index_info["info"]["fields"] == ["title", "TEXT"]
    _assert_no_sensitive_markers(
        {
            "slowlog": slowlog,
            "config": config,
            "clients": clients,
            "memory": memory,
            "sample": sample,
            "indexes": indexes,
            "index_info": index_info,
        }
    )


@pytest.mark.asyncio
async def test_command_errors_return_redacted_error_payload() -> None:
    provider = make_provider(
        FakeRedisClient(info_error=RuntimeError(f"cannot connect to {_URL}"))
    )

    result = await provider.info()

    assert result["status"] == "error"
    assert "error" in result
    _assert_no_sensitive_markers(result)


@pytest.mark.asyncio
async def test_memory_stats_marks_unsupported_command() -> None:
    provider = make_provider(
        FakeRedisClient(memory_stats_error=RuntimeError("unknown command 'MEMORY'"))
    )

    result = await provider.memory_stats()

    assert result["status"] == "error"
    assert result["error_type"] == "unsupported_command"


def test_get_client_uses_real_url_but_logs_only_masked_url(monkeypatch, caplog) -> None:
    fake_client = FakeRedisClient()
    seen: dict[str, Any] = {}

    def fake_from_url(url: str, *, decode_responses: bool) -> FakeRedisClient:
        seen["url"] = url
        seen["decode_responses"] = decode_responses
        return fake_client

    monkeypatch.setattr(provider_module.Redis, "from_url", fake_from_url)
    caplog.set_level(logging.INFO)
    provider = RedisCommandToolProvider(redis_instance=make_instance())

    assert provider.get_client() is fake_client
    assert seen == {"url": _URL, "decode_responses": True}
    _assert_no_sensitive_markers(caplog.text)
