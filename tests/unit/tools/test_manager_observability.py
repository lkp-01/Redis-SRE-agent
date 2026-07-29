"""Redis、Prometheus 和 Loki 接入 ToolManager 的离线契约测试。"""

from __future__ import annotations

import hashlib
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from redis_sre_agent.core.config import Settings
from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.tools.manager import ToolManager
from redis_sre_agent.tools.logs.loki import provider as loki_provider_module
from redis_sre_agent.tools.logs.loki.provider import LokiToolProvider
from redis_sre_agent.tools.metrics.prometheus import provider as provider_module
from redis_sre_agent.tools.metrics.prometheus.provider import PrometheusToolProvider
from redis_sre_agent.tools.models import ToolCapability


_REDIS_PROVIDER = (
    "redis_sre_agent.tools.diagnostics.redis_command.provider.RedisCommandToolProvider"
)
_PROMETHEUS_PROVIDER = (
    "redis_sre_agent.tools.metrics.prometheus.provider.PrometheusToolProvider"
)
_LOKI_PROVIDER = "redis_sre_agent.tools.logs.loki.provider.LokiToolProvider"
_EXPECTED_REDIS_OPERATIONS = {
    "info",
    "slowlog",
    "acl_log",
    "config_get",
    "client_list",
    "cluster_info",
    "replication_info",
    "memory_stats",
    "sample_keys",
    "search_indexes",
    "search_index_info",
}
_EXPECTED_PROMETHEUS_OPERATIONS = {"query", "query_range", "search_metrics"}
_EXPECTED_LOKI_OPERATIONS = {
    "query",
    "query_range",
    "labels",
    "label_values",
    "series",
    "volume",
    "patterns",
}


def _make_instance() -> RedisInstance:
    return RedisInstance(
        id="observability-target",
        name="Observability Target",
        connection_url=SecretStr("redis://localhost:6379/0"),
        environment="test",
        usage="diagnostics",
        description="Manager observability contract target",
    )


def test_default_provider_order_is_redis_then_prometheus_then_loki() -> None:
    with patch.dict(os.environ, {}, clear=True):
        config = Settings(_env_file=None)

    assert config.tool_providers == [
        _REDIS_PROVIDER,
        _PROMETHEUS_PROVIDER,
        _LOKI_PROVIDER,
    ]
    assert all(
        excluded not in " ".join(config.tool_providers).lower()
        for excluded in ("host_telemetry", "support_package", "redis_cloud", "re_admin")
    )


@pytest.mark.asyncio
async def test_manager_registers_21_observability_tools_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import redis_sre_agent.tools.manager as manager_module

    config = Settings(
        _env_file=None,
        rag_enabled=False,
        mcp_servers={},
        tool_providers=[_REDIS_PROVIDER, _PROMETHEUS_PROVIDER, _LOKI_PROVIDER],
    )
    monkeypatch.setattr(manager_module, "settings", config)

    def forbidden_external_call(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("构造 ToolManager 时禁止访问 Prometheus 或 Loki")

    prometheus_constructor = MagicMock(side_effect=forbidden_external_call)
    monkeypatch.setattr(provider_module, "PrometheusConnect", prometheus_constructor)
    monkeypatch.setattr(provider_module.requests, "get", forbidden_external_call)
    loki_client = MagicMock(side_effect=forbidden_external_call)
    monkeypatch.setattr(loki_provider_module.httpx, "AsyncClient", loki_client)

    async def fake_query(
        self: PrometheusToolProvider, query: str
    ) -> dict[str, Any]:
        return {"status": "success", "query": query, "data": []}

    monkeypatch.setattr(PrometheusToolProvider, "query", fake_query)

    async def fake_loki_query(
        self: LokiToolProvider,
        query: str,
        time: str | None = None,
        limit: int | None = None,
        direction: str | None = None,
    ) -> dict[str, Any]:
        return {"status": "success", "query": query, "data": []}

    monkeypatch.setattr(LokiToolProvider, "query", fake_loki_query)
    instance = _make_instance()
    instance_hash = hashlib.sha256(instance.id.encode()).hexdigest()[:6]

    async with ToolManager(redis_instance=instance) as manager:
        redis_definitions = manager.get_tools_by_provider_names(["redis_command"])
        prometheus_definitions = manager.get_tools_by_provider_names(["prometheus"])
        loki_definitions = manager.get_tools_by_provider_names(["loki"])
        redis_provider = next(
            provider
            for provider in manager.get_providers_for_capability(
                ToolCapability.DIAGNOSTICS
            )
            if provider.provider_name == "redis_command"
        )
        prometheus_provider = next(
            provider
            for provider in manager.get_providers_for_capability(
                ToolCapability.METRICS
            )
            if provider.provider_name == "prometheus"
        )
        loki_provider = next(
            provider
            for provider in manager.get_providers_for_capability(ToolCapability.LOGS)
            if provider.provider_name == "loki"
        )

        redis_operations = {
            redis_provider.resolve_operation(tool.name, {})
            for tool in redis_definitions
        }
        prometheus_operations = {
            prometheus_provider.resolve_operation(tool.name, {})
            for tool in prometheus_definitions
        }
        loki_operations = {
            loki_provider.resolve_operation(tool.name, {})
            for tool in loki_definitions
        }
        query_tool = next(
            tool.name
            for tool in prometheus_definitions
            if prometheus_provider.resolve_operation(tool.name, {}) == "query"
        )
        routed_result = await manager.resolve_tool_call(
            query_tool,
            {"query": "up"},
        )
        loki_query_tool = next(
            tool.name
            for tool in loki_definitions
            if loki_provider.resolve_operation(tool.name, {}) == "query"
        )
        loki_routed_result = await manager.resolve_tool_call(
            loki_query_tool,
            {"query": '{job="redis"}'},
        )
        excluded_definitions = manager.get_tools_by_provider_names(
            ["host_telemetry", "support_package", "redis_cloud", "re_admin"]
        )

    assert redis_operations == _EXPECTED_REDIS_OPERATIONS
    assert prometheus_operations == _EXPECTED_PROMETHEUS_OPERATIONS
    assert loki_operations == _EXPECTED_LOKI_OPERATIONS
    assert len(redis_definitions) == 11
    assert len(prometheus_definitions) == 3
    assert len(loki_definitions) == 7
    assert len(redis_definitions + prometheus_definitions + loki_definitions) == 21
    assert all(
        tool.name.startswith(f"redis_command_{instance_hash}_")
        for tool in redis_definitions
    )
    assert all(
        tool.name.startswith(f"prometheus_{instance_hash}_")
        for tool in prometheus_definitions
    )
    assert all(
        tool.name.startswith(f"loki_{instance_hash}_")
        for tool in loki_definitions
    )
    assert routed_result == {"status": "success", "query": "up", "data": []}
    assert loki_routed_result == {
        "status": "success",
        "query": '{job="redis"}',
        "data": [],
    }
    assert excluded_definitions == []
    assert prometheus_provider._client is None
    prometheus_constructor.assert_not_called()
    loki_client.assert_not_called()
