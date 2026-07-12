"""RAG 向量器配置隔离测试；全部使用 fake，不访问模型或网络。"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from redis_sre_agent.core.config import Settings
from redis_sre_agent.core import vectorizer_helpers


class FakeVectorizer:
    async def aembed(self, *_args: Any, **_kwargs: Any) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def aembed_many(self, values: list[str], **_kwargs: Any) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in values]


def test_openai_vectorizer_uses_embedding_credentials_only(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeOpenAIVectorizer(FakeVectorizer):
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(vectorizer_helpers, "_build_embeddings_cache", lambda _config: object())
    monkeypatch.setattr(
        "redis_sre_agent.core.redis.OpenAITextVectorizer",
        FakeOpenAIVectorizer,
    )

    config = Settings(
        _env_file=None,
        rag_enabled=True,
        openai_api_key=SecretStr("CHAT_KEY_MUST_NOT_BE_USED"),
        openai_base_url="https://chat.deepseek.invalid/v1",
        embedding_api_key=SecretStr("EMBEDDING_KEY_FOR_TEST"),
        embedding_base_url="https://embedding.example.invalid/v1",
        embedding_provider="openai",
        embedding_model="embedding-test-model",
        vector_dim=3,
    )

    result = vectorizer_helpers.create_vectorizer(config=config)

    assert isinstance(result, FakeOpenAIVectorizer)
    assert captured["model"] == "embedding-test-model"
    assert captured["api_config"] == {
        "api_key": "EMBEDDING_KEY_FOR_TEST",
        "base_url": "https://embedding.example.invalid/v1",
    }
    assert "CHAT_KEY_MUST_NOT_BE_USED" not in repr(captured)
    assert "deepseek" not in repr(captured).lower()


def test_chat_credentials_never_satisfy_embedding_configuration(monkeypatch) -> None:
    monkeypatch.setattr(vectorizer_helpers, "_build_embeddings_cache", lambda _config: object())
    config = Settings(
        _env_file=None,
        rag_enabled=True,
        openai_api_key=SecretStr("CHAT_ONLY_SECRET"),
        openai_base_url="https://api.deepseek.com",
        embedding_api_key=None,
        embedding_base_url=None,
        embedding_provider="openai",
    )

    with pytest.raises(ValueError, match="embedding_api_key") as exc_info:
        vectorizer_helpers.create_vectorizer(config=config)

    assert "CHAT_ONLY_SECRET" not in str(exc_info.value)
    assert "deepseek" not in str(exc_info.value).lower()


def test_custom_factory_receives_embedding_only_config(monkeypatch) -> None:
    received: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeVectorizer:
        received.update(kwargs)
        return FakeVectorizer()

    monkeypatch.setattr(vectorizer_helpers, "_build_embeddings_cache", lambda _config: object())
    vectorizer_helpers.set_vectorizer_factory(factory)
    try:
        config = Settings(
            _env_file=None,
            rag_enabled=True,
            openai_api_key=SecretStr("CHAT_SECRET"),
            openai_base_url="https://api.deepseek.com",
            vectorizer_factory="tests.fake.embedding_factory",
            embedding_provider="custom",
            embedding_model="custom-model",
            vector_dim=3,
        )

        result = vectorizer_helpers.create_vectorizer(config=config)
    finally:
        vectorizer_helpers.set_vectorizer_factory(None)

    assert isinstance(result, FakeVectorizer)
    factory_config = received["config"]
    assert factory_config.embedding_provider == "custom"
    assert factory_config.embedding_model == "custom-model"
    assert factory_config.vector_dim == 3
    assert factory_config.vectorizer_factory == "tests.fake.embedding_factory"
    assert not hasattr(factory_config, "openai_api_key")
    assert not hasattr(factory_config, "openai_base_url")
    assert not hasattr(factory_config, "redis_url")


def test_factory_result_must_support_both_async_embedding_methods(monkeypatch) -> None:
    class IncompleteVectorizer:
        async def aembed(self, *_args: Any, **_kwargs: Any) -> list[float]:
            return [0.1]

    monkeypatch.setattr(vectorizer_helpers, "_build_embeddings_cache", lambda _config: object())
    vectorizer_helpers.set_vectorizer_factory(lambda **_kwargs: IncompleteVectorizer())
    try:
        config = Settings(
            _env_file=None,
            rag_enabled=True,
            vectorizer_factory="tests.fake.embedding_factory",
            embedding_provider="custom",
        )
        with pytest.raises(TypeError, match="aembed_many"):
            vectorizer_helpers.create_vectorizer(config=config)
    finally:
        vectorizer_helpers.set_vectorizer_factory(None)
