"""DeepSeek LLM 工厂测试；所有模型调用都使用本地 stub，不访问网络。"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from redis_sre_agent.core import llm_helpers


class StubModel:
    def __init__(self, *, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[Any] = []
        self.bound_tools: list[Any] | None = None

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        self.calls.append((messages, kwargs))
        if self.error is not None:
            raise self.error
        return self.result

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        self.calls.append((messages, kwargs))
        if self.error is not None:
            raise self.error
        return self.result

    def bind_tools(self, tools: Any, **_: Any) -> "StubModel":
        bound = StubModel(result=self.result, error=self.error)
        bound.bound_tools = list(tools)
        return bound


def configure_deepseek(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_helpers.settings, "openai_api_key", SecretStr("unit-test-key"))
    monkeypatch.setattr(llm_helpers.settings, "openai_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(llm_helpers.settings, "openai_model", "deepseek-v4-pro")
    monkeypatch.setattr(llm_helpers.settings, "openai_model_mini", "deepseek-v4-flash")
    monkeypatch.setattr(llm_helpers.settings, "openai_model_nano", "deepseek-v4-flash")
    monkeypatch.setattr(llm_helpers.settings, "llm_timeout", 180.0)
    monkeypatch.setattr(llm_helpers.settings, "llm_failover_enabled", True)
    monkeypatch.setattr(llm_helpers.settings, "deepseek_thinking_mode", "disabled")


def test_create_llm_builds_deepseek_primary_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_deepseek(monkeypatch)

    model = llm_helpers.create_llm()

    assert isinstance(model, llm_helpers.FailoverChatModel)
    assert model.primary.model_name == "deepseek-v4-pro"
    assert model.fallback.model_name == "deepseek-v4-flash"
    assert model.primary.openai_api_base == "https://api.deepseek.com"
    assert model.primary.request_timeout == 180.0
    assert model.primary.extra_body == {"thinking": {"type": "disabled"}}


def test_create_mini_and_nano_use_flash_without_nested_failover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_deepseek(monkeypatch)

    mini = llm_helpers.create_mini_llm()
    nano = llm_helpers.create_nano_llm()

    assert not isinstance(mini, llm_helpers.FailoverChatModel)
    assert not isinstance(nano, llm_helpers.FailoverChatModel)
    assert mini.model_name == "deepseek-v4-flash"
    assert nano.model_name == "deepseek-v4-flash"


@pytest.mark.asyncio
async def test_failover_retries_only_the_failed_model_call() -> None:
    primary = StubModel(error=RuntimeError("primary unavailable"))
    fallback = StubModel(result="flash response")
    model = llm_helpers.FailoverChatModel(primary=primary, fallback=fallback)

    result = await model.ainvoke(["same messages"])

    assert result == "flash response"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1
    assert primary.calls[0][0] is fallback.calls[0][0]


@pytest.mark.asyncio
async def test_bound_failover_binds_both_models_to_the_same_tools() -> None:
    primary = StubModel(error=RuntimeError("primary unavailable"))
    fallback = StubModel(result="tool-aware response")

    bound = llm_helpers.FailoverChatModel(primary=primary, fallback=fallback).bind_tools(["info"])
    result = await bound.ainvoke(["diagnose"])

    assert result == "tool-aware response"
    assert bound.primary.bound_tools == ["info"]
    assert bound.fallback.bound_tools == ["info"]


@pytest.mark.asyncio
async def test_failover_error_does_not_expose_provider_error_text() -> None:
    secret_marker = "unit-test-secret-marker"
    model = llm_helpers.FailoverChatModel(
        primary=StubModel(error=RuntimeError(f"primary {secret_marker}")),
        fallback=StubModel(error=RuntimeError(f"fallback {secret_marker}")),
    )

    with pytest.raises(llm_helpers.LLMFailoverError) as exc_info:
        await model.ainvoke(["diagnose"])

    rendered = str(exc_info.value)
    assert secret_marker not in rendered
    assert "RuntimeError" in rendered
