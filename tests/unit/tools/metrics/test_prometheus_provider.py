"""Prometheus provider 契约测试；所有外部请求都使用 mock。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import requests

from redis_sre_agent.tools.metrics.prometheus import provider as provider_module
from redis_sre_agent.tools.metrics.prometheus.provider import (
    PrometheusConfig,
    PrometheusToolProvider,
)
from redis_sre_agent.tools.models import ToolCapability


_TEST_URL = "http://prometheus.test:9090"
_EXPECTED_SCHEMA = {
    "query": ({"query"}, {"query"}),
    "query_range": (
        {"query", "start_time", "end_time", "step"},
        {"query", "start_time"},
    ),
    "search_metrics": ({"pattern", "label_filters"}, set()),
}
_SECRET_URL = "http://user:prometheus-secret@prometheus.test:9090"


def _make_provider() -> PrometheusToolProvider:
    config = PrometheusConfig(
        url=_TEST_URL,
        disable_ssl=False,
        _env_file=None,
    )
    return PrometheusToolProvider(config=config)


def _assert_secret_free(value: Any) -> None:
    rendered = repr(value)
    assert "prometheus-secret" not in rendered
    assert _SECRET_URL not in rendered


@pytest.fixture(autouse=True)
def _block_real_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """任何漏掉的 requests mock 都应立即让测试失败，而不是访问网络。"""

    def forbidden_request(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Prometheus 单元测试禁止真实 HTTP 请求")

    monkeypatch.setattr(requests, "get", forbidden_request)


def test_config_defaults_and_environment_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as context:
        context.delenv("TOOLS_PROMETHEUS_URL", raising=False)
        context.delenv("TOOLS_PROMETHEUS_DISABLE_SSL", raising=False)
        default_config = PrometheusConfig(_env_file=None)

    assert default_config.url == "http://localhost:9090"
    assert default_config.disable_ssl is False

    monkeypatch.setenv("TOOLS_PROMETHEUS_URL", "http://custom.test:19090")
    monkeypatch.setenv("TOOLS_PROMETHEUS_DISABLE_SSL", "true")
    env_config = PrometheusConfig(_env_file=None)

    assert env_config.url == "http://custom.test:19090"
    assert env_config.disable_ssl is True


def test_provider_exposes_exact_metrics_schema_contract() -> None:
    provider = _make_provider()
    schemas = provider.create_tool_schemas()
    by_operation = {
        provider.resolve_operation(schema.name, {}): schema for schema in schemas
    }

    assert provider.provider_name == "prometheus"
    assert provider.requires_redis_instance is False
    assert set(by_operation) == set(_EXPECTED_SCHEMA)
    assert len(schemas) == 3
    for operation, (properties, required) in _EXPECTED_SCHEMA.items():
        schema = by_operation[operation]
        assert schema.capability is ToolCapability.METRICS
        assert schema.parameters["type"] == "object"
        assert set(schema.parameters["properties"]) == properties
        assert set(schema.parameters["required"]) == required


def test_client_is_lazy_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr(provider_module, "PrometheusConnect", constructor)

    provider = _make_provider()

    assert provider._client is None
    constructor.assert_not_called()
    assert provider.get_client() is client
    assert provider.get_client() is client
    constructor.assert_called_once_with(url=_TEST_URL, disable_ssl=False)


@pytest.mark.asyncio
async def test_tools_bind_each_schema_to_its_provider_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()
    method_args = {
        "query": {"query": "up"},
        "query_range": {"query": "up", "start_time": "1h"},
        "search_metrics": {"pattern": "redis"},
    }
    method_mocks: dict[str, AsyncMock] = {}
    for operation in _EXPECTED_SCHEMA:
        method_mock = AsyncMock(return_value={"operation": operation})
        monkeypatch.setattr(provider, operation, method_mock)
        method_mocks[operation] = method_mock

    tools = provider.tools()
    by_operation = {
        provider.resolve_operation(tool.definition.name, {}): tool for tool in tools
    }

    assert set(by_operation) == set(_EXPECTED_SCHEMA)
    for operation, args in method_args.items():
        result = await by_operation[operation].invoke(args)
        assert result == {"operation": operation}
        method_mocks[operation].assert_awaited_once_with(**args)


@pytest.mark.asyncio
async def test_query_returns_successful_instant_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()
    wait_for_targets = AsyncMock()
    http_get = AsyncMock(
        return_value={
            "status": "success",
            "data": {
                "result": [
                    {"metric": {"__name__": "redis_up"}, "value": [1_700_000_000, "1"]}
                ]
            },
        }
    )
    monkeypatch.setattr(provider, "_wait_for_targets", wait_for_targets)
    monkeypatch.setattr(provider, "_http_get_json", http_get)

    result = await provider.query("redis_up")

    assert result["status"] == "success"
    assert result["query"] == "redis_up"
    assert len(result["data"]) == 1
    wait_for_targets.assert_awaited_once_with(timeout_seconds=10.0)
    http_get.assert_awaited_once_with("/api/v1/query", params={"query": "redis_up"})


@pytest.mark.asyncio
async def test_query_returns_success_with_empty_data_after_bounded_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()
    http_get = AsyncMock(
        return_value={"status": "success", "data": {"result": []}}
    )
    monkeypatch.setattr(provider, "_wait_for_targets", AsyncMock())
    monkeypatch.setattr(provider, "_http_get_json", http_get)
    monkeypatch.setattr(provider_module.asyncio, "sleep", AsyncMock())

    result = await provider.query("redis_connected_clients")

    assert result["status"] == "success"
    assert result["data"] == []
    assert http_get.await_count == 6


@pytest.mark.asyncio
async def test_query_localizes_api_and_exception_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()
    monkeypatch.setattr(provider, "_wait_for_targets", AsyncMock())
    monkeypatch.setattr(
        provider,
        "_http_get_json",
        AsyncMock(return_value={"status": "error", "error": "invalid PromQL"}),
    )

    api_error = await provider.query("invalid{")

    assert api_error == {
        "status": "error",
        "error": "invalid PromQL",
        "query": "invalid{",
    }

    monkeypatch.setattr(
        provider,
        "_wait_for_targets",
        AsyncMock(side_effect=RuntimeError(f"cannot connect to {_SECRET_URL}")),
    )
    exception_error = await provider.query("up")

    assert exception_error["status"] == "error"
    _assert_secret_free(exception_error)


@pytest.mark.asyncio
async def test_query_range_returns_successful_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 1, tzinfo=timezone.utc)
    parse_datetime = MagicMock(side_effect=[start, end])
    http_get = AsyncMock(
        return_value={
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {"__name__": "redis_up"},
                        "values": [[int(start.timestamp()), "1"]],
                    }
                ]
            },
        }
    )
    monkeypatch.setattr(provider_module, "parse_datetime", parse_datetime)
    monkeypatch.setattr(provider, "_wait_for_targets", AsyncMock())
    monkeypatch.setattr(provider, "_http_get_json", http_get)

    result = await provider.query_range("redis_up", "1h", "now", "30s")

    assert result["status"] == "success"
    assert len(result["data"]) == 1
    http_get.assert_awaited_once_with(
        "/api/v1/query_range",
        params={
            "query": "redis_up",
            "start": int(start.timestamp()),
            "end": int(end.timestamp()),
            "step": "30s",
        },
    )


@pytest.mark.asyncio
async def test_query_range_returns_empty_data_after_bounded_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 1, tzinfo=timezone.utc)
    http_get = AsyncMock(
        return_value={"status": "success", "data": {"result": []}}
    )
    monkeypatch.setattr(
        provider_module,
        "parse_datetime",
        MagicMock(side_effect=[start, end]),
    )
    monkeypatch.setattr(provider, "_wait_for_targets", AsyncMock())
    monkeypatch.setattr(provider, "_http_get_json", http_get)
    monkeypatch.setattr(provider_module.asyncio, "sleep", AsyncMock())

    result = await provider.query_range(
        "redis_connected_clients", "1h", "now", "15s"
    )

    assert result["status"] == "success"
    assert result["data"] == []
    assert http_get.await_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_position", ["start", "end"])
async def test_query_range_rejects_invalid_times(
    invalid_position: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()
    valid = datetime(2026, 1, 1, tzinfo=timezone.utc)
    parsed = [None, valid] if invalid_position == "start" else [valid, None]
    http_get = AsyncMock()
    monkeypatch.setattr(
        provider_module,
        "parse_datetime",
        MagicMock(side_effect=parsed),
    )
    monkeypatch.setattr(provider, "_wait_for_targets", AsyncMock())
    monkeypatch.setattr(provider, "_http_get_json", http_get)

    result = await provider.query_range("redis_up", "bad-start", "bad-end")

    assert result["status"] == "error"
    assert f"Invalid {invalid_position}_time" in result["error"]
    http_get.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_range_localizes_and_redacts_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()
    monkeypatch.setattr(provider, "_wait_for_targets", AsyncMock())
    monkeypatch.setattr(
        provider_module,
        "parse_datetime",
        MagicMock(side_effect=RuntimeError(f"cannot connect to {_SECRET_URL}")),
    )

    result = await provider.query_range("redis_up", "1h", "now")

    assert result["status"] == "error"
    _assert_secret_free(result)


@pytest.mark.asyncio
async def test_http_helper_uses_thread_boundary_and_redacts_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()
    response = MagicMock()
    response.json.return_value = {"status": "success", "data": {"result": []}}
    to_thread = AsyncMock(return_value=response)
    monkeypatch.setattr(provider_module.asyncio, "to_thread", to_thread)

    success = await provider._http_get_json(
        "/api/v1/query", params={"query": "up"}
    )

    assert success["status"] == "success"
    to_thread.assert_awaited_once()

    monkeypatch.setattr(
        provider_module.asyncio,
        "to_thread",
        AsyncMock(side_effect=RuntimeError(f"cannot connect to {_SECRET_URL}")),
    )
    error = await provider._http_get_json("/api/v1/query")

    assert error["status"] == "error"
    _assert_secret_free(error)


@pytest.mark.asyncio
async def test_wait_for_targets_uses_thread_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "status": "success",
        "data": {"activeTargets": [{"labels": {"job": "redis"}}]},
    }
    to_thread = AsyncMock(return_value=response)
    monkeypatch.setattr(provider_module.asyncio, "to_thread", to_thread)

    await provider._wait_for_targets(timeout_seconds=0.1)

    to_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_metrics_filters_pattern_and_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()

    async def fake_http_get(
        path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if path == "/api/v1/label/__name__/values":
            return {
                "status": "success",
                "data": ["redis_up", "redis_commands_total", "process_cpu_seconds_total"],
            }
        assert path == "/api/v1/query"
        query = str((params or {}).get("query", ""))
        result = [{"metric": {"job": "redis"}}] if query.startswith("redis_up{") else []
        return {"status": "success", "data": {"result": result}}

    http_get = AsyncMock(side_effect=fake_http_get)
    monkeypatch.setattr(provider, "_http_get_json", http_get)

    result = await provider.search_metrics(
        pattern="REDIS", label_filters={"job": "redis"}
    )

    assert result["status"] == "success"
    assert result["metrics"] == ["redis_up"]
    assert result["count"] == 1
    assert result["label_filters"] == {"job": "redis"}
    queried = [call.kwargs.get("params", {}).get("query", "") for call in http_get.await_args_list]
    assert 'redis_up{job="redis"}' in queried


@pytest.mark.asyncio
async def test_search_metrics_localizes_and_redacts_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()
    monkeypatch.setattr(
        provider,
        "_http_get_json",
        AsyncMock(side_effect=RuntimeError(f"cannot connect to {_SECRET_URL}")),
    )

    result = await provider.search_metrics("redis")

    assert result["status"] == "error"
    _assert_secret_free(result)
