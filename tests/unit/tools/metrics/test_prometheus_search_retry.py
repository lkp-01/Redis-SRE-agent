"""Prometheus 指标发现重试测试；验证重试仍留在线程边界内。"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from redis_sre_agent.tools.metrics.prometheus import provider as provider_module
from redis_sre_agent.tools.metrics.prometheus.provider import (
    PrometheusConfig,
    PrometheusToolProvider,
)


def _make_provider() -> PrometheusToolProvider:
    return PrometheusToolProvider(
        config=PrometheusConfig(
            url="http://prometheus.test:9090",
            disable_ssl=False,
            _env_file=None,
        )
    )


@pytest.mark.asyncio
async def test_search_metrics_http_retry_returns_second_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()
    http_get = AsyncMock(
        side_effect=[
            {"status": "success", "data": []},
            {"status": "success", "data": ["redis_up", "prometheus_build_info"]},
        ]
    )
    get_client = MagicMock()
    monkeypatch.setattr(provider, "_http_get_json", http_get)
    monkeypatch.setattr(provider, "get_client", get_client)
    monkeypatch.setattr(provider_module.asyncio, "sleep", AsyncMock())

    result = await provider.search_metrics(pattern="redis")

    assert result["status"] == "success"
    assert result["metrics"] == ["redis_up"]
    assert http_get.await_count == 2
    get_client.assert_not_called()


@pytest.mark.asyncio
async def test_search_metrics_client_retry_uses_thread_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()
    client = MagicMock()
    client.all_metrics = MagicMock(
        side_effect=[[], ["redis_up", "prometheus_build_info"]]
    )
    monkeypatch.setattr(provider, "get_client", MagicMock(return_value=client))
    monkeypatch.setattr(
        provider,
        "_http_get_json",
        AsyncMock(return_value={"status": "success", "data": []}),
    )
    monkeypatch.setattr(provider_module.asyncio, "sleep", AsyncMock())

    async def run_in_thread(
        function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return function(*args, **kwargs)

    to_thread = AsyncMock(side_effect=run_in_thread)
    monkeypatch.setattr(provider_module.asyncio, "to_thread", to_thread)

    result = await provider.search_metrics(pattern="redis")

    assert result["status"] == "success"
    assert result["metrics"] == ["redis_up"]
    assert client.all_metrics.call_count == 2
    assert to_thread.await_count == 2
