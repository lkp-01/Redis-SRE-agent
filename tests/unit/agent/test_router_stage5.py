"""Stage 5 router 边界测试。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from redis_sre_agent.agent import router as router_module
from redis_sre_agent.agent.router import AgentType, query_needs_live_redis_scope, route_to_appropriate_agent
from redis_sre_agent.core import llm_helpers


class RouterLLM:
    def __init__(self, *, content: str = "CHAT", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls = 0

    async def ainvoke(self, _messages) -> AIMessage:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.content)


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


@pytest.mark.asyncio
async def test_query_needs_live_redis_scope_understands_chinese_intent() -> None:
    assert await query_needs_live_redis_scope("Redis 有点慢，帮我看看咋回事") is True
    assert await query_needs_live_redis_scope("排查一下 Redis 延迟") is True
    assert await query_needs_live_redis_scope("解释一下 Redis 主从复制原理") is False
    assert await query_needs_live_redis_scope("Redis 最佳实践有哪些") is False


@pytest.mark.asyncio
async def test_router_uses_nano_factory_when_key_is_configured(monkeypatch) -> None:
    llm = RouterLLM(content="DEEP_TRIAGE")
    monkeypatch.setattr(router_module.settings, "openai_api_key", SecretStr("configured"))
    monkeypatch.setattr(llm_helpers, "create_nano_llm", lambda timeout=10.0: llm)

    route = await route_to_appropriate_agent("analyze Redis")

    assert route is AgentType.REDIS_TRIAGE
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_router_model_failure_keeps_deterministic_fallback(monkeypatch) -> None:
    llm = RouterLLM(error=RuntimeError("provider unavailable"))
    monkeypatch.setattr(router_module.settings, "openai_api_key", SecretStr("configured"))
    monkeypatch.setattr(llm_helpers, "create_nano_llm", lambda timeout=10.0: llm)

    route = await route_to_appropriate_agent("deep triage Redis")

    assert route is AgentType.REDIS_TRIAGE
    assert llm.calls == 1
