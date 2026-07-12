"""RAG disabled/not-ready/ready 三态与显式索引责任测试。"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from pydantic import SecretStr

from redis_sre_agent.cli.main import main
from redis_sre_agent.core.config import Settings
from redis_sre_agent.core import redis as redis_core


def _ft_info(schema: dict[str, Any]) -> list[Any]:
    attributes: list[list[Any]] = []
    for field in schema["fields"]:
        field_type = str(field["type"]).upper()
        attribute: list[Any] = [
            b"identifier",
            field["name"].encode(),
            b"attribute",
            field["name"].encode(),
            b"type",
            field_type.encode(),
        ]
        if field_type == "VECTOR":
            attrs = field["attrs"]
            attribute.extend(
                [
                    b"algorithm",
                    str(attrs["algorithm"]).upper().encode(),
                    b"data_type",
                    str(attrs["datatype"]).upper().encode(),
                    b"dim",
                    int(attrs["dims"]),
                    b"distance_metric",
                    str(attrs["distance_metric"]).upper().encode(),
                ]
            )
        attributes.append(attribute)
    return [b"attributes", attributes]


class FakeSearchClient:
    def __init__(self, *, search_available: bool, schema: dict[str, Any]) -> None:
        self.search_available = search_available
        self.schema = schema
        self.commands: list[tuple[Any, ...]] = []

    async def execute_command(self, *args: Any) -> Any:
        self.commands.append(args)
        command = " ".join(str(part).upper() for part in args[:2])
        if command == "COMMAND INFO":
            if self.search_available:
                return [{"name": "ft.search"}, {"name": "ft.create"}]
            return [None, None]
        if str(args[0]).upper() == "FT.INFO":
            return _ft_info(self.schema)
        if str(args[0]).upper() == "FT.SEARCH":
            if not self.search_available:
                raise RuntimeError("unknown command")
            return [0]
        raise AssertionError(f"unexpected command: {args!r}")


class FakeKnowledgeIndex:
    def __init__(
        self,
        *,
        exists: bool,
        search_available: bool = True,
        schema: dict[str, Any],
    ) -> None:
        self._exists = exists
        self.create_calls = 0
        self._redis_client = FakeSearchClient(
            search_available=search_available,
            schema=schema,
        )
        self.client = self._redis_client

    async def exists(self) -> bool:
        return self._exists

    async def create(self) -> None:
        self.create_calls += 1
        self._exists = True


def _enabled_config(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "rag_enabled": True,
        "embedding_provider": "openai",
        "embedding_api_key": SecretStr("EMBEDDING_SECRET_FOR_TEST"),
        "embedding_base_url": "https://embedding.example.invalid/v1",
        "vector_dim": 3,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.asyncio
async def test_disabled_readiness_does_not_touch_embedding_or_redis(monkeypatch) -> None:
    async def forbidden_index(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("disabled RAG must not construct the knowledge index")

    monkeypatch.setattr(redis_core, "get_knowledge_index", forbidden_index)
    monkeypatch.setattr(
        "redis_sre_agent.core.vectorizer_helpers.validate_embedding_config",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("disabled RAG must not inspect embedding configuration")
        ),
    )

    readiness = await redis_core.get_rag_readiness(
        Settings(_env_file=None, rag_enabled=False)
    )

    assert readiness.state == "disabled"
    assert readiness.reason_code == "disabled"
    assert readiness.ready is False


@pytest.mark.asyncio
async def test_embedding_invalid_is_not_ready_and_does_not_reuse_chat_secret(monkeypatch) -> None:
    async def forbidden_index(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("invalid embedding config must fail before Redis access")

    monkeypatch.setattr(redis_core, "get_knowledge_index", forbidden_index)
    config = Settings(
        _env_file=None,
        rag_enabled=True,
        openai_api_key=SecretStr("CHAT_SECRET_MUST_STAY_PRIVATE"),
        openai_base_url="https://api.deepseek.com",
        embedding_provider="openai",
        embedding_api_key=None,
    )

    readiness = await redis_core.get_rag_readiness(config)

    assert readiness.state == "not_ready"
    assert readiness.reason_code == "embedding_config_invalid"
    rendered = str(readiness.as_dict())
    assert "CHAT_SECRET_MUST_STAY_PRIVATE" not in rendered
    assert "deepseek" not in rendered.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("search_available", "exists", "actual_dim", "expected_reason"),
    [
        (False, False, 3, "redis_search_unavailable"),
        (True, False, 3, "index_missing"),
        (True, True, 4, "schema_mismatch"),
        (True, True, 3, "ready"),
    ],
)
async def test_readiness_reason_codes(
    monkeypatch,
    search_available: bool,
    exists: bool,
    actual_dim: int,
    expected_reason: str,
) -> None:
    config = _enabled_config(vector_dim=3)
    actual_schema = redis_core._build_document_schema(
        redis_core.SRE_KNOWLEDGE_INDEX,
        include_pinned=True,
        vector_dim=actual_dim,
    )
    index = FakeKnowledgeIndex(
        exists=exists,
        search_available=search_available,
        schema=actual_schema,
    )

    async def fake_get_index(_config=None):
        return index

    monkeypatch.setattr(redis_core, "get_knowledge_index", fake_get_index)

    readiness = await redis_core.get_rag_readiness(config)

    assert readiness.reason_code == expected_reason
    assert readiness.state == ("ready" if expected_reason == "ready" else "not_ready")
    assert readiness.ready is (expected_reason == "ready")
    assert index.create_calls == 0


@pytest.mark.asyncio
async def test_get_knowledge_index_only_constructs_and_never_creates(monkeypatch) -> None:
    created: dict[str, Any] = {}

    class FakeAsyncSearchIndex:
        def __init__(self, *, schema: Any, redis_client: Any) -> None:
            created["schema"] = schema
            created["redis_client"] = redis_client
            self.create_calls = 0

        async def create(self) -> None:
            self.create_calls += 1

    fake_client = object()
    fake_schema = object()
    monkeypatch.setattr(redis_core, "get_redis_client", lambda **_kwargs: fake_client)
    monkeypatch.setattr(redis_core, "AsyncSearchIndex", FakeAsyncSearchIndex)
    monkeypatch.setattr(redis_core.IndexSchema, "from_dict", Mock(return_value=fake_schema))

    index = await redis_core.get_knowledge_index(_enabled_config())

    assert index.create_calls == 0
    assert created == {"schema": fake_schema, "redis_client": fake_client}


@pytest.mark.asyncio
async def test_only_explicit_ensure_can_create_missing_index(monkeypatch) -> None:
    config = _enabled_config()
    schema = redis_core._build_document_schema(
        redis_core.SRE_KNOWLEDGE_INDEX,
        include_pinned=True,
        vector_dim=config.vector_dim,
    )
    index = FakeKnowledgeIndex(exists=False, schema=schema)

    async def fake_get_index(_config=None):
        return index

    monkeypatch.setattr(redis_core, "get_knowledge_index", fake_get_index)

    with pytest.raises(redis_core.RAGNotReadyError) as exc_info:
        await redis_core.ensure_knowledge_index(config, create_if_missing=False)
    assert exc_info.value.reason_code == "index_missing"
    assert index.create_calls == 0

    ensured = await redis_core.ensure_knowledge_index(config, create_if_missing=True)

    assert ensured is index
    assert index.create_calls == 1
    assert any(str(command[0]).upper() == "FT.SEARCH" for command in index.client.commands)


def test_status_cli_reports_safe_rag_state(monkeypatch) -> None:
    readiness = redis_core.RAGReadiness(
        state="not_ready",
        reason_code="index_missing",
        message="knowledge index is missing",
    )

    async def fake_readiness(_config=None):
        return readiness

    monkeypatch.setattr(redis_core, "get_rag_readiness", fake_readiness)

    result = CliRunner().invoke(main, ["status"])

    assert result.exit_code == 0, result.output
    assert "not_ready" in result.output
    assert "index_missing" in result.output
    assert "redis://" not in result.output

