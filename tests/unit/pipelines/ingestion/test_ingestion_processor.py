"""直接 Markdown 摄取的原子替换、失败保旧与确定性重试测试。"""

from __future__ import annotations

import fnmatch
import inspect
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from redis_sre_agent.core.config import Settings
from redis_sre_agent.pipelines.ingestion._processor_impl import IngestionPipeline


class FakeTransactionalPipeline:
    def __init__(self, redis: "FakeTransactionalRedis") -> None:
        self.redis = redis
        self.commands: list[tuple[str, Any]] = []

    def delete(self, *keys: str) -> "FakeTransactionalPipeline":
        self.commands.append(("delete", keys))
        return self

    def hset(self, key: str, *, mapping: dict[str, Any]) -> "FakeTransactionalPipeline":
        self.commands.append(("hset", (key, deepcopy(mapping))))
        return self

    async def execute(self) -> list[Any]:
        if self.redis.fail_next_transaction:
            self.redis.fail_next_transaction = False
            raise RuntimeError("simulated transaction failure")
        staged = deepcopy(self.redis.hashes)
        results: list[Any] = []
        for command, payload in self.commands:
            if command == "delete":
                deleted = 0
                for key in payload:
                    deleted += int(staged.pop(str(key), None) is not None)
                results.append(deleted)
            elif command == "hset":
                key, mapping = payload
                staged[str(key)] = deepcopy(mapping)
                results.append(len(mapping))
            else:  # pragma: no cover - fake contract guard
                raise AssertionError(command)
        self.redis.hashes = staged
        self.redis.commit_count += 1
        return results


class FakeTransactionalRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, Any]] = {}
        self.fail_next_transaction = False
        self.commit_count = 0

    async def hgetall(self, key: str) -> dict[str, Any]:
        return deepcopy(self.hashes.get(str(key), {}))

    async def scan_iter(self, match: str):
        for key in sorted(self.hashes):
            if fnmatch.fnmatch(key, match):
                yield key

    def pipeline(self, *, transaction: bool) -> FakeTransactionalPipeline:
        assert transaction is True
        return FakeTransactionalPipeline(self)


class FakeIndex:
    def __init__(self, client: FakeTransactionalRedis) -> None:
        self.client = client
        self._redis_client = client


class FakeVectorizer:
    def __init__(self, *, dim: int = 3) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []
        self.fail = False

    async def aembed(self, value: str, **_kwargs: Any) -> list[float]:
        return (await self.aembed_many([value]))[0]

    async def aembed_many(self, values: list[str], **_kwargs: Any) -> list[list[float]]:
        self.calls.append(list(values))
        if self.fail:
            raise RuntimeError("simulated embedding failure")
        return [[float(index + 1) for index in range(self.dim)] for _ in values]


def _write_markdown(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "title: Memory runbook\n"
        "category: shared\n"
        "doc_type: runbook\n"
        "version: latest\n"
        "---\n"
        f"# Memory runbook\n\n{body}\n",
        encoding="utf-8",
    )


def _pipeline(
    redis: FakeTransactionalRedis,
    vectorizer: FakeVectorizer,
) -> IngestionPipeline:
    return IngestionPipeline(
        settings_config=Settings(
            _env_file=None,
            rag_enabled=True,
            embedding_provider="custom",
            vectorizer_factory="tests.fake.vectorizer",
            vector_dim=3,
        ),
        index=FakeIndex(redis),
        vectorizer=vectorizer,
    )


def _source_tracking(redis: FakeTransactionalRedis) -> tuple[str, dict[str, Any]]:
    matches = [key for key in redis.hashes if key.startswith("sre_knowledge_meta:source:")]
    assert len(matches) == 1
    key = matches[0]
    return key, redis.hashes[key]


@pytest.mark.asyncio
async def test_add_and_repeat_are_atomic_and_idempotent(tmp_path: Path) -> None:
    source_dir = tmp_path / "source_documents"
    _write_markdown(source_dir / "shared" / "memory.md", "Inspect INFO memory first.")
    redis = FakeTransactionalRedis()
    vectorizer = FakeVectorizer()
    pipeline = _pipeline(redis, vectorizer)

    first = await pipeline.ingest_source_documents(source_dir)
    commits_after_first = redis.commit_count
    embedding_calls_after_first = len(vectorizer.calls)
    second = await pipeline.ingest_source_documents(source_dir)

    assert first[0]["status"] == "success"
    assert first[0]["action"] == "add"
    assert first[0]["chunks_indexed"] == 1
    assert second[0]["action"] == "unchanged"
    assert redis.commit_count == commits_after_first == 1
    assert len(vectorizer.calls) == embedding_calls_after_first == 1
    _, tracking = _source_tracking(redis)
    document_hash = tracking["document_hash"]
    assert f"sre_knowledge:{document_hash}:chunk:0" in redis.hashes
    assert f"sre_knowledge_meta:{document_hash}" in redis.hashes


@pytest.mark.asyncio
async def test_embedding_failure_makes_zero_redis_mutations_and_keeps_old_version(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source_documents"
    source = source_dir / "shared" / "memory.md"
    _write_markdown(source, "Old, still searchable guidance.")
    redis = FakeTransactionalRedis()
    vectorizer = FakeVectorizer()
    pipeline = _pipeline(redis, vectorizer)
    await pipeline.ingest_source_documents(source_dir)
    _, old_tracking = _source_tracking(redis)
    old_hash = old_tracking["document_hash"]
    snapshot = deepcopy(redis.hashes)
    commits = redis.commit_count

    _write_markdown(source, "New guidance that cannot be embedded yet.")
    vectorizer.fail = True
    result = await pipeline.ingest_source_documents(source_dir)

    assert result[0]["status"] == "error"
    assert redis.hashes == snapshot
    assert redis.commit_count == commits
    assert f"sre_knowledge:{old_hash}:chunk:0" in redis.hashes


@pytest.mark.asyncio
async def test_transaction_failure_keeps_old_version_and_same_input_retry_recovers(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source_documents"
    source = source_dir / "shared" / "memory.md"
    _write_markdown(source, "Old searchable guidance.")
    redis = FakeTransactionalRedis()
    vectorizer = FakeVectorizer()
    pipeline = _pipeline(redis, vectorizer)
    await pipeline.ingest_source_documents(source_dir)
    source_key, old_tracking = _source_tracking(redis)
    old_hash = old_tracking["document_hash"]
    snapshot = deepcopy(redis.hashes)

    _write_markdown(source, "New replacement guidance.")
    redis.fail_next_transaction = True
    failed = await pipeline.ingest_source_documents(source_dir)

    assert failed[0]["status"] == "error"
    assert redis.hashes == snapshot
    assert redis.hashes[source_key]["document_hash"] == old_hash
    assert f"sre_knowledge:{old_hash}:chunk:0" in redis.hashes

    retried = await pipeline.ingest_source_documents(source_dir)
    _, new_tracking = _source_tracking(redis)
    new_hash = new_tracking["document_hash"]

    assert retried[0]["status"] == "success"
    assert retried[0]["action"] == "update"
    assert new_hash != old_hash
    assert f"sre_knowledge:{new_hash}:chunk:0" in redis.hashes
    assert f"sre_knowledge:{old_hash}:chunk:0" not in redis.hashes
    assert f"sre_knowledge_meta:{old_hash}" not in redis.hashes


@pytest.mark.asyncio
async def test_vector_dimension_failure_occurs_before_transaction(tmp_path: Path) -> None:
    source_dir = tmp_path / "source_documents"
    _write_markdown(source_dir / "shared" / "memory.md", "Dimension mismatch.")
    redis = FakeTransactionalRedis()
    vectorizer = FakeVectorizer(dim=2)
    pipeline = _pipeline(redis, vectorizer)

    result = await pipeline.ingest_source_documents(source_dir)

    assert result[0]["status"] == "error"
    assert redis.hashes == {}
    assert redis.commit_count == 0


def test_processor_has_no_skills_hidden_import() -> None:
    import redis_sre_agent.pipelines.ingestion._processor_impl as implementation

    source = inspect.getsource(implementation)
    assert "redis_sre_agent.skills" not in source
