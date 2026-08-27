"""Eval Redis 环境物化和真实工具链测试。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from redis.asyncio import Redis

from evals import utils as eval_utils
from evals.utils import EvalEnvironment, EvalRuntime, materialize_environment, run_agent_async
from redis_sre_agent.agent.models import AgentExecutionTrace, AgentResponse
from redis_sre_agent.tools.diagnostics.redis_command.provider import RedisCommandToolProvider
from redis_sre_agent.tools.manager import ToolManager
from redis_sre_agent.tools.models import ToolCapability


pytestmark = [
    pytest.mark.eval_category("environment"),
    pytest.mark.eval_tier("baseline"),
]


class _InjectionSpyAgent:
    def __init__(self, redis_instance: object, redis_cluster: object) -> None:
        self.redis_instance = redis_instance
        self.redis_cluster = redis_cluster
        self.seen_redis_instance: object | None = None
        self.seen_redis_cluster: object | None = None

    async def process_query(self, query: str, **_kwargs: object) -> AgentResponse:
        self.seen_redis_instance = self.redis_instance
        self.seen_redis_cluster = self.redis_cluster
        return AgentResponse(
            response=f"processed: {query}",
            trace=AgentExecutionTrace(
                messages=[
                    HumanMessage(content=query),
                    AIMessage(content="done"),
                ],
                iteration_count=1,
            ),
        )


class _FailingInjectionSpyAgent(_InjectionSpyAgent):
    async def process_query(self, query: str, **_kwargs: object) -> AgentResponse:
        self.seen_redis_instance = self.redis_instance
        self.seen_redis_cluster = self.redis_cluster
        raise RuntimeError(f"agent failed for {query}")


@pytest.mark.asyncio
async def test_run_agent_async_injects_eval_runtime_and_restores_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_instance = object()
    original_cluster = object()
    runtime_instance = object()
    runtime = SimpleNamespace(redis_instance=runtime_instance, redis_url="redis://127.0.0.1:6379/15")
    agent = _InjectionSpyAgent(original_instance, original_cluster)

    @asynccontextmanager
    async def fake_materialize(_environment: EvalEnvironment):
        yield runtime

    monkeypatch.setattr(eval_utils, "materialize_environment", fake_materialize)

    trajectory = await run_agent_async(
        agent,
        query="inspect the eval target",
        environment=EvalEnvironment(redis_data={}),
        model=SimpleNamespace(model_name="eval-test-model"),
    )

    assert agent.seen_redis_instance is runtime_instance
    assert agent.seen_redis_cluster is None
    assert agent.redis_instance is original_instance
    assert agent.redis_cluster is original_cluster
    assert trajectory.answer == "processed: inspect the eval target"


@pytest.mark.asyncio
async def test_run_agent_async_restores_agent_after_agent_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_instance = object()
    original_cluster = object()
    runtime_instance = object()
    runtime = SimpleNamespace(redis_instance=runtime_instance, redis_url="redis://127.0.0.1:6379/15")
    agent = _FailingInjectionSpyAgent(original_instance, original_cluster)

    @asynccontextmanager
    async def fake_materialize(_environment: EvalEnvironment):
        yield runtime

    monkeypatch.setattr(eval_utils, "materialize_environment", fake_materialize)

    with pytest.raises(RuntimeError, match="agent failed"):
        await run_agent_async(
            agent,
            query="inspect the eval target",
            environment=EvalEnvironment(redis_data={}),
            model=SimpleNamespace(model_name="eval-test-model"),
        )

    assert agent.seen_redis_instance is runtime_instance
    assert agent.seen_redis_cluster is None
    assert agent.redis_instance is original_instance
    assert agent.redis_cluster is original_cluster


@pytest.mark.asyncio
async def test_materialized_environment_is_visible_through_real_redis_tools(
    eval_redis_url: str,
) -> None:
    key = "eval:real-tool-chain:unique-key"
    environment = EvalEnvironment(redis_data={key: "present"})

    async with materialize_environment(environment, redis_url=eval_redis_url) as runtime:
        assert runtime.redis_instance.connection_url.get_secret_value() == eval_redis_url

        async with ToolManager(redis_instance=runtime.redis_instance) as manager:
            providers = manager.get_providers_for_capability(ToolCapability.DIAGNOSTICS)
            redis_provider = next(
                provider
                for provider in providers
                if isinstance(provider, RedisCommandToolProvider)
            )
            sample_keys_tool = next(
                tool
                for tool in manager.get_tools_by_provider_names(["redis_command"])
                if redis_provider.resolve_operation(tool.name, {}) == "sample_keys"
            )

            result = await manager.resolve_tool_call(sample_keys_tool.name, {"count": 10})

    assert result["status"] == "success"
    assert any(item["key"] == key for item in result["keys"])

    client = Redis.from_url(eval_redis_url, decode_responses=True)
    try:
        assert await client.exists(key) == 0
    finally:
        await client.aclose()
