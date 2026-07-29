"""显式 opt-in 的真实 Prometheus 只读 smoke。"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
import requests


_LIVE_ENABLED = os.getenv("RUN_PROMETHEUS_INTEGRATION", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_PROMETHEUS_URL = os.getenv("TOOLS_PROMETHEUS_URL", "").strip()
_DISABLE_SSL = os.getenv("TOOLS_PROMETHEUS_DISABLE_SSL", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_LIVE_ENABLED and _PROMETHEUS_URL),
        reason=(
            "需要 RUN_PROMETHEUS_INTEGRATION=1 和 TOOLS_PROMETHEUS_URL 才运行真实 Prometheus smoke"
        ),
    ),
]

from redis_sre_agent.tools.metrics.prometheus.provider import (  # noqa: E402
    PrometheusConfig,
    PrometheusToolProvider,
)


@pytest.fixture
def prometheus_provider() -> PrometheusToolProvider:
    config = PrometheusConfig(
        url=_PROMETHEUS_URL,
        disable_ssl=_DISABLE_SSL,
        _env_file=None,
    )
    return PrometheusToolProvider(config=config)


@pytest.mark.asyncio
async def test_prometheus_ready_and_instant_queries(
    prometheus_provider: PrometheusToolProvider,
) -> None:
    try:
        response = await asyncio.to_thread(
            requests.get,
            f"{_PROMETHEUS_URL.rstrip('/')}/-/ready",
            timeout=5,
            verify=not _DISABLE_SSL,
        )
    except Exception:
        pytest.fail("显式配置的 Prometheus readiness 地址不可用", pytrace=False)

    assert response.status_code == 200

    up_result = await prometheus_provider.query("up")
    redis_result = await prometheus_provider.query("redis_up")

    assert up_result["status"] == "success"
    assert up_result["data"]
    assert redis_result["status"] == "success"
    assert redis_result["data"]


@pytest.mark.asyncio
async def test_prometheus_range_and_metric_discovery(
    prometheus_provider: PrometheusToolProvider,
) -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=5)

    range_result = await prometheus_provider.query_range(
        "up",
        start_time=start.isoformat(),
        end_time=end.isoformat(),
        step="15s",
    )
    search_result = await prometheus_provider.search_metrics("redis")

    assert range_result["status"] == "success"
    assert range_result["data"]
    assert search_result["status"] == "success"
    assert "redis_up" in search_result["metrics"]
