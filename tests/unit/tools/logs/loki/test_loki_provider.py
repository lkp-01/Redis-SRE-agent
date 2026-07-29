"""Loki provider 的离线契约测试，所有 HTTP 调用都由 mock 接管。"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.tools.logs.loki.provider import (
    LokiConfig,
    LokiToolProvider,
)
from redis_sre_agent.tools.models import ToolActionKind, ToolCapability


_EXPECTED_OPERATIONS = {
    "query",
    "query_range",
    "labels",
    "label_values",
    "series",
    "volume",
    "patterns",
}


def _make_instance(extension_data: dict[str, Any] | None = None) -> RedisInstance:
    return RedisInstance(
        id="loki-test-target",
        name="Loki Test Target",
        connection_url=SecretStr("redis://localhost:6379/0"),
        environment="test",
        usage="diagnostics",
        description="Loki provider contract target",
        extension_data=extension_data,
    )


def _mock_http_client(response: MagicMock | None = None, error: Exception | None = None):
    client = AsyncMock()
    if error is not None:
        client.request = AsyncMock(side_effect=error)
    else:
        client.request = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return patch("httpx.AsyncClient", return_value=client), client


def test_config_reads_environment_and_builds_tenant_header() -> None:
    with patch.dict(
        os.environ,
        {
            "TOOLS_LOKI_URL": "http://loki:3100",
            "TOOLS_LOKI_TENANT_ID": "tenant-a",
            "TOOLS_LOKI_TIMEOUT": "12.5",
            "TOOLS_LOKI_DEFAULT_SELECTOR": '{job="redis"}',
        },
        clear=True,
    ):
        provider = LokiToolProvider(config=LokiConfig(_env_file=None))

    assert provider.config.url == "http://loki:3100"
    assert provider.config.timeout == 12.5
    assert provider.config.default_selector == '{job="redis"}'
    assert provider._headers() == {
        "Accept": "application/json",
        "X-Scope-OrgID": "tenant-a",
    }


def test_default_config_has_no_tenant_header() -> None:
    with patch.dict(os.environ, {}, clear=True):
        provider = LokiToolProvider(config=LokiConfig(_env_file=None))

    assert provider.config.url == "http://localhost:3100"
    assert provider._headers() == {"Accept": "application/json"}


def test_provider_exposes_exactly_seven_read_only_log_tools() -> None:
    provider = LokiToolProvider(redis_instance=_make_instance())
    tools = provider.tools()
    operations = {
        provider.resolve_operation(tool.metadata.name, {}) for tool in tools
    }

    assert operations == _EXPECTED_OPERATIONS
    assert provider.capabilities == {ToolCapability.LOGS}
    assert all(tool.metadata.capability is ToolCapability.LOGS for tool in tools)
    assert all(tool.metadata.action_kind is ToolActionKind.READ for tool in tools)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1705312800", "1705312800000000000"),
        ("1705312800000", "1705312800000000000"),
        ("1705312800000000", "1705312800000000000"),
        ("1705312800000000000", "1705312800000000000"),
        ("2024-01-15T10:00:00Z", "1705312800000000000"),
    ],
)
def test_time_parser_normalizes_supported_precisions(value: str, expected: str) -> None:
    provider = LokiToolProvider()

    assert provider._parse_time_to_epoch_ns(value) == expected


def test_time_parser_supports_now_and_relative_duration() -> None:
    provider = LokiToolProvider()
    before = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    now_value = int(provider._parse_time_to_epoch_ns("now") or 0)
    one_hour_ago = int(provider._parse_time_to_epoch_ns("1h") or 0)
    after = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)

    assert before <= now_value <= after
    assert 3_599_000_000_000 <= now_value - one_hour_ago <= 3_601_000_000_000


@pytest.mark.asyncio
async def test_empty_selector_uses_instance_streams_and_default_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _make_instance(
        {
            "loki": {
                "prefer_streams": [{"job": "redis", "instance": "primary"}],
                "default_selector": '{service="redis-primary"}',
            }
        }
    )
    provider = LokiToolProvider(
        redis_instance=instance,
        config=LokiConfig(default_selector='{redis_role=~".+"}'),
    )
    request = AsyncMock(return_value={"status": "success", "code": 200, "data": {}})
    monkeypatch.setattr(provider, "_request", request)

    await provider.query(query='{} |= "Background saving"')

    sent_query = request.await_args.kwargs["params"]["query"]
    assert '({job="redis",instance="primary"} |= "Background saving")' in sent_query
    assert '({service="redis-primary"} |= "Background saving")' in sent_query
    assert '({redis_role=~".+"} |= "Background saving")' in sent_query


@pytest.mark.asyncio
async def test_empty_selector_has_safe_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LokiToolProvider(config=LokiConfig(default_selector=None))
    request = AsyncMock(return_value={"status": "success", "code": 200, "data": {}})
    monkeypatch.setattr(provider, "_request", request)

    await provider.query(query='{} |= "error"')

    sent_query = request.await_args.kwargs["params"]["query"]
    assert sent_query == '({job=~".+"} |= "error") or ({service=~".+"} |= "error")'


@pytest.mark.asyncio
async def test_non_empty_selector_is_not_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LokiToolProvider(config=LokiConfig(default_selector='{job="redis"}'))
    request = AsyncMock(return_value={"status": "success", "code": 200, "data": {}})
    monkeypatch.setattr(provider, "_request", request)
    query = '{job="redis"} |= "error"'

    await provider.query(query=query)

    assert request.await_args.kwargs["params"]["query"] == query


@pytest.mark.asyncio
async def test_all_operations_use_expected_loki_api_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LokiToolProvider(config=LokiConfig(default_selector='{job="redis"}'))
    request = AsyncMock(return_value={"status": "success", "code": 200, "data": {}})
    monkeypatch.setattr(provider, "_request", request)

    await provider.query('{job="redis"}', time="1705312800", limit=10, direction="backward")
    await provider.query_range('{job="redis"}', "1705312800", "1705312860", step="1s")
    await provider.labels(start="1705312800", end="1705312860", query='{job="redis"}')
    await provider.label_values("service", start="1705312800", end="1705312860")
    await provider.series(['{job="redis"}'], start="1705312800", end="1705312860")
    await provider.volume(
        '{job="redis"}',
        "1705312800",
        "1705312860",
        limit=5,
        targetLabels="service",
        aggregateBy="labels",
    )
    await provider.patterns('{job="redis"}', "1705312800", "1705312860", step="1s")

    paths = [call.args[1] for call in request.await_args_list]
    assert paths == [
        "/loki/api/v1/query",
        "/loki/api/v1/query_range",
        "/loki/api/v1/labels",
        "/loki/api/v1/label/service/values",
        "/loki/api/v1/series",
        "/loki/api/v1/index/volume",
        "/loki/api/v1/patterns",
    ]
    series_call = request.await_args_list[4]
    assert series_call.args[0] == "POST"
    assert series_call.kwargs["data"] == {"match[]": ['{job="redis"}']}
    volume_params = request.await_args_list[5].kwargs["params"]
    assert volume_params["targetLabels"] == "service"
    assert volume_params["aggregateBy"] == "labels"


@pytest.mark.asyncio
async def test_request_success_uses_timeout_and_headers() -> None:
    response = MagicMock(
        status_code=200,
        headers={"content-type": "application/json; charset=utf-8"},
    )
    response.json.return_value = {"status": "success", "data": {"result": []}}
    provider = LokiToolProvider(config=LokiConfig(tenant_id="tenant-a", timeout=9.0))
    context, client = _mock_http_client(response=response)

    with context as client_class:
        result = await provider._request("GET", "/loki/api/v1/query", params={"query": "up"})

    assert result == {
        "status": "success",
        "code": 200,
        "data": {"status": "success", "data": {"result": []}},
    }
    client_class.assert_called_once_with(
        timeout=9.0,
        headers={"Accept": "application/json", "X-Scope-OrgID": "tenant-a"},
    )
    client.request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 500])
async def test_request_http_errors_are_structured_and_redacted(status_code: int) -> None:
    response = MagicMock(
        status_code=status_code,
        headers={"content-type": "application/json"},
    )
    response.json.return_value = {
        "message": "token=super-secret at http://user:pass@loki:3100/path?api_key=hidden"
    }
    provider = LokiToolProvider()
    context, _client = _mock_http_client(response=response)

    with context:
        result = await provider._request("GET", "/loki/api/v1/query")

    rendered = repr(result)
    assert result["status"] == "error"
    assert result["code"] == status_code
    assert "super-secret" not in rendered
    assert "user:pass" not in rendered
    assert "api_key=hidden" not in rendered


@pytest.mark.asyncio
async def test_request_non_json_error_is_structured() -> None:
    response = MagicMock(status_code=502, headers={"content-type": "text/plain"})
    response.text = "upstream unavailable"
    provider = LokiToolProvider()
    context, _client = _mock_http_client(response=response)

    with context:
        result = await provider._request("GET", "/loki/api/v1/labels")

    assert result == {
        "status": "error",
        "code": 502,
        "error": {"raw": "upstream unavailable"},
    }


@pytest.mark.asyncio
async def test_request_connection_error_is_structured_and_redacted() -> None:
    provider = LokiToolProvider()
    error = httpx.ConnectError(
        "failed http://user:pass@loki:3100/path?token=super-secret"
    )
    context, _client = _mock_http_client(error=error)

    with context:
        result = await provider._request("GET", "/loki/api/v1/query")

    assert result["status"] == "error"
    assert "super-secret" not in result["error"]
    assert "user:pass" not in result["error"]
    assert "loki:3100/path" in result["error"]
