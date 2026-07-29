"""Prometheus 只读查询 provider。"""

from redis_sre_agent.tools.metrics.prometheus.provider import (
    PrometheusConfig,
    PrometheusToolProvider,
)

__all__ = ["PrometheusConfig", "PrometheusToolProvider"]
