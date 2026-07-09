"""Stage 5 router 边界测试。"""

from __future__ import annotations

import pytest

from redis_sre_agent.agent.router import AgentType, query_needs_live_redis_scope, route_to_appropriate_agent


@pytest.mark.asyncio
async def test_router_defaults_regular_redis_query_to_chat() -> None:
    route = await route_to_appropriate_agent("check Redis memory")

    assert route is AgentType.REDIS_CHAT


@pytest.mark.asyncio
async def test_router_uses_triage_for_explicit_deep_requests() -> None:
    assert await route_to_appropriate_agent("deep triage Redis") is AgentType.REDIS_TRIAGE
    assert await route_to_appropriate_agent("comprehensive analysis Redis") is AgentType.REDIS_TRIAGE
    assert await route_to_appropriate_agent("exhaustive Redis investigation") is AgentType.REDIS_TRIAGE


@pytest.mark.asyncio
async def test_router_uses_triage_for_support_package_context() -> None:
    route = await route_to_appropriate_agent(
        "check this package",
        context={"support_package_path": "support-package-slot"},
    )

    assert route is AgentType.REDIS_TRIAGE


@pytest.mark.asyncio
async def test_query_needs_live_redis_scope_is_deterministic() -> None:
    assert await query_needs_live_redis_scope("diagnose Redis memory") is True
    assert await query_needs_live_redis_scope("show slowlog and clients") is True
    assert await query_needs_live_redis_scope("what are Redis best practices?") is False
