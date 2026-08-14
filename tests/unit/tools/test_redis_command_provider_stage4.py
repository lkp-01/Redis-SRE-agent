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
_EXPECTED_OPERATIONS = {
    "info",
    "slowlog",
    "acl_log",
    "config_get",
    "client_list",
    "cluster_info",
    "replication_info",
    "memory_stats",
    "bigkey_scan",
    "sample_keys",
    "search_indexes",
    "search_index_info",
}


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

    def memory_usage(self, key: str, samples: int = 5) -> None:
        self.commands.append(("memory_usage", (key, samples)))

    async def execute(self) -> list[Any]:
        results: list[Any] = []
        random_index = 0
        for operation, argument in self.commands:
            if operation == "randomkey":
                results.append(
                    self.client.random_keys[random_index % len(self.client.random_keys)]
                )
                random_index += 1
            elif operation == "type":
                results.append(self.client.key_types.get(str(argument), "string"))
            elif operation == "memory_usage":
                if self.client.memory_usage_error is not None:
                    raise self.client.memory_usage_error
                key, _samples = argument
                results.append(self.client.memory_sizes.get(str(key)))
        return results


class FakeRedisClient:
    def __init__(
        self,
        *,
        info_error: Exception | None = None,
        memory_stats_error: Exception | None = None,
        memory_usage_error: Exception | None = None,
    ) -> None:
        self.info_error = info_error
        self.memory_stats_error = memory_stats_error
        self.memory_usage_error = memory_usage_error
        self.closed = False
        self.random_keys = ["session:1", "cart:1", "session:1", "stream:1", None]
        self.key_types = {
            "session:1": "string",
            "cart:1": "hash",
            "stream:1": "stream",
        }
        self.memory_sizes = {
            "session:1": 512,
            "cart:1": 2_500_000,
            "stream:1": 1_500_000,
        }
        self.scan_calls = 0

    async def scan(
        self,
        cursor: int = 0,
        *,
        count: int | None = None,
    ) -> tuple[int, list[str]]:
        self.scan_calls += 1
        return 0, ["session:1", "cart:1", "stream:1"]

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
    schemas = provider.create_tool_schemas()
    operations = {provider.resolve_operation(tool.name, {}) for tool in schemas}

    assert len(schemas) == len(_EXPECTED_OPERATIONS)
    assert operations == _EXPECTED_OPERATIONS


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
    bigkeys = await provider.bigkey_scan(
        threshold_bytes=1_000_000,
        max_keys=10,
        scan_count=10,
        top_n=2,
        time_limit_ms=1_000,
    )
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
    assert bigkeys["status"] == "success"
    assert bigkeys["scan_complete"] is True
    assert bigkeys["stop_reason"] == "cursor_exhausted"
    assert bigkeys["keys_scanned"] == 3
    assert bigkeys["keys_measured"] == 3
    assert bigkeys["big_key_count"] == 2
    assert [item["key"] for item in bigkeys["largest_keys"]] == [
        "cart:1",
        "stream:1",
    ]
    assert all(item["is_big"] for item in bigkeys["big_keys"])
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
            "bigkeys": bigkeys,
            "sample": sample,
            "indexes": indexes,
            "index_info": index_info,
        }
    )


@pytest.mark.asyncio
async def test_bigkey_scan_stops_at_key_budget_without_claiming_full_scan() -> None:
    client = FakeRedisClient()

    async def paged_scan(
        cursor: int = 0,
        *,
        count: int | None = None,
    ) -> tuple[int, list[str]]:
        client.scan_calls += 1
        return 7, ["session:1", "cart:1", "stream:1"]

    client.scan = paged_scan  # type: ignore[method-assign]
    provider = make_provider(client)

    result = await provider.bigkey_scan(
        threshold_bytes=1_000_000,
        max_keys=2,
        scan_count=100,
        top_n=10,
        time_limit_ms=1_000,
    )

    assert result["status"] == "success"
    assert result["scan_complete"] is False
    assert result["stop_reason"] == "max_keys_reached"
    assert result["keys_scanned"] == 2
    assert result["keys_measured"] == 2
    assert client.scan_calls == 1


@pytest.mark.asyncio
async def test_bigkey_scan_marks_memory_usage_as_unsupported() -> None:
    provider = make_provider(
        FakeRedisClient(memory_usage_error=RuntimeError("unknown command 'MEMORY'"))
    )

    result = await provider.bigkey_scan()

    assert result["status"] == "error"
    assert result["error_type"] == "unsupported_command"


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


@pytest.mark.asyncio
async def test_optional_cluster_and_search_commands_return_error_envelopes() -> None:
    class UnsupportedFeatureClient(FakeRedisClient):
        async def cluster(self, subcommand: str) -> Any:
            raise RuntimeError("This instance has cluster support disabled")

        async def execute_command(self, command: str, *args: Any) -> Any:
            raise RuntimeError(f"unknown command {command!r}")

    provider = make_provider(UnsupportedFeatureClient())

    results = [
        await provider.cluster_info(),
        await provider.search_indexes(),
        await provider.search_index_info("idx:missing"),
    ]

    assert all(result["status"] == "error" for result in results)
    assert all("error" in result for result in results)
    _assert_no_sensitive_markers(results)


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

    assert provider._client is None
    assert provider.get_client() is fake_client
    assert seen == {"url": _URL, "decode_responses": True}
    _assert_no_sensitive_markers(caplog.text)
