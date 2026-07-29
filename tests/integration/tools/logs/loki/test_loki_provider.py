"""显式 opt-in 的真实 Loki 只读 smoke。"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest


_LIVE_ENABLED = os.getenv("RUN_LOKI_INTEGRATION", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_LOKI_URL = os.getenv("TOOLS_LOKI_URL", "").strip()
_LOKI_TENANT_ID = os.getenv("TOOLS_LOKI_TENANT_ID", "").strip() or None
_DEFAULT_SELECTOR = (
    os.getenv("TOOLS_LOKI_DEFAULT_SELECTOR", "").strip() or '{job="redis"}'
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_LIVE_ENABLED and _LOKI_URL),
        reason="需要 RUN_LOKI_INTEGRATION=1 和 TOOLS_LOKI_URL 才运行真实 Loki smoke",
    ),
]

from redis_sre_agent.tools.logs.loki.provider import (  # noqa: E402
    LokiConfig,
    LokiToolProvider,
)


@pytest.fixture
def loki_provider() -> LokiToolProvider:
    return LokiToolProvider(
        config=LokiConfig(
            url=_LOKI_URL,
            tenant_id=_LOKI_TENANT_ID,
            default_selector=_DEFAULT_SELECTOR,
            _env_file=None,
        )
    )


def _loki_data(result: dict[str, Any]) -> Any:
    assert result["status"] == "success"
    payload = result["data"]
    assert payload["status"] == "success"
    return payload["data"]


@pytest.mark.asyncio
async def test_loki_ready_labels_values_and_series(
    loki_provider: LokiToolProvider,
) -> None:
    headers = loki_provider._headers()
    try:
        async with httpx.AsyncClient(timeout=5, headers=headers) as client:
            response = await client.get(f"{_LOKI_URL.rstrip('/')}/ready")
    except Exception:
        pytest.fail("显式配置的 Loki readiness 地址不可用", pytrace=False)

    assert response.status_code == 200

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=6)
    labels = _loki_data(
        await loki_provider.labels(start=start.isoformat(), end=end.isoformat())
    )
    job_values = _loki_data(
        await loki_provider.label_values(
            "job",
            start=start.isoformat(),
            end=end.isoformat(),
            query=_DEFAULT_SELECTOR,
        )
    )
    series = _loki_data(
        await loki_provider.series(
            [_DEFAULT_SELECTOR],
            start=start.isoformat(),
            end=end.isoformat(),
        )
    )

    assert {"job", "service", "instance"}.issubset(set(labels))
    assert "redis" in job_values
    assert series


@pytest.mark.asyncio
async def test_loki_range_and_instant_queries(
    loki_provider: LokiToolProvider,
) -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=6)

    range_data = _loki_data(
        await loki_provider.query_range(
            _DEFAULT_SELECTOR,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=50,
            direction="backward",
        )
    )
    instant_data = _loki_data(
        await loki_provider.query(
            f"sum(count_over_time({_DEFAULT_SELECTOR}[6h]))",
            time=end.isoformat(),
        )
    )

    assert range_data["result"]
    assert instant_data["result"]


@pytest.mark.asyncio
async def test_loki_volume_and_patterns(
    loki_provider: LokiToolProvider,
) -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=6)

    volume_data = _loki_data(
        await loki_provider.volume(
            _DEFAULT_SELECTOR,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=20,
            target_labels="service",
            aggregate_by="labels",
        )
    )
    patterns_data = _loki_data(
        await loki_provider.patterns(
            _DEFAULT_SELECTOR,
            start=start.isoformat(),
            end=end.isoformat(),
            step="5s",
        )
    )

    assert volume_data["result"]
    assert isinstance(patterns_data, list)
