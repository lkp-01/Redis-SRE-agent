"""ArtifactStorage、batch manifest 与 prepared batch 流程测试。"""

from __future__ import annotations

import fnmatch
import inspect
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from redis_sre_agent.core.config import Settings
from redis_sre_agent.pipelines.ingestion.processor import IngestionPipeline
from redis_sre_agent.pipelines.scraper.base import (
    ArtifactStorage,
    DocumentCategory,
    DocumentType,
    ScrapedDocument,
    SeverityLevel,
)


class FakePipeline:
    def __init__(self, redis: "FakeRedis") -> None:
        self.redis = redis
        self.commands: list[tuple[str, Any]] = []

    def delete(self, *keys: str) -> "FakePipeline":
        self.commands.append(("delete", keys))
        return self

    def hset(self, key: str, *, mapping: dict[str, Any]) -> "FakePipeline":
        self.commands.append(("hset", (key, deepcopy(mapping))))
        return self

    async def execute(self) -> list[int]:
        staged = deepcopy(self.redis.hashes)
        results = []
        for command, payload in self.commands:
            if command == "delete":
                for key in payload:
                    staged.pop(str(key), None)
                results.append(1)
            else:
                key, mapping = payload
                staged[str(key)] = mapping
                results.append(len(mapping))
        self.redis.hashes = staged
        return results


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, Any]] = {}

    async def hgetall(self, key: str) -> dict[str, Any]:
        return deepcopy(self.hashes.get(str(key), {}))

    async def scan_iter(self, match: str):
        for key in sorted(self.hashes):
            if fnmatch.fnmatch(key, match):
                yield key

    def pipeline(self, *, transaction: bool) -> FakePipeline:
        assert transaction is True
        return FakePipeline(self)


class FakeIndex:
    def __init__(self, client: FakeRedis) -> None:
        self.client = client
        self._redis_client = client


class FakeVectorizer:
    async def aembed(self, _value: str, **_kwargs: Any) -> list[float]:
        return [1.0, 0.0, 0.0]

    async def aembed_many(self, values: list[str], **_kwargs: Any) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in values]


def _document() -> ScrapedDocument:
    return ScrapedDocument(
        title="Memory runbook",
        content="# Memory\n\nInspect INFO memory.",
        source_url="file://shared/memory.md",
        category=DocumentCategory.SHARED,
        doc_type=DocumentType.RUNBOOK,
        severity=SeverityLevel.HIGH,
        metadata={
            "source_document_path": "shared/memory.md",
            "source_document_scope": "",
            "version": "latest",
        },
    )


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        rag_enabled=True,
        embedding_provider="custom",
        vectorizer_factory="tests.fake.vectorizer",
        vector_dim=3,
    )


def test_artifact_json_round_trip_preserves_document_contract(tmp_path: Path) -> None:
    storage = ArtifactStorage(tmp_path / "artifacts")
    storage.set_batch_date("2026-07-12")
    original = _document()

    path = storage.save_document(original)
    restored = ScrapedDocument.from_dict(json.loads(path.read_text(encoding="utf-8")))

    assert restored.to_dict()["title"] == original.title
    assert restored.content == original.content
    assert restored.source_url == original.source_url
    assert restored.category is DocumentCategory.SHARED
    assert restored.doc_type is DocumentType.RUNBOOK
    assert restored.severity is SeverityLevel.HIGH
    assert restored.document_hash == original.document_hash


def test_batch_manifest_summarizes_stored_artifacts(tmp_path: Path) -> None:
    storage = ArtifactStorage(tmp_path / "artifacts")
    storage.set_batch_date("2026-07-12")
    storage.save_document(_document())

    manifest_path = storage.save_batch_manifest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["batch_date"] == "2026-07-12"
    assert manifest["total_documents"] == 1
    assert manifest["categories"] == {"shared": 1}
    assert manifest["document_types"] == {"runbook": 1}
    assert manifest["sources"] == ["file://shared/memory.md"]
    assert storage.list_available_batches() == ["2026-07-12"]
    assert storage.get_batch_manifest("2026-07-12") == manifest


@pytest.mark.asyncio
async def test_prepare_source_artifacts_then_ingest_prepared_batch(tmp_path: Path) -> None:
    source_dir = tmp_path / "source_documents"
    markdown = source_dir / "shared" / "memory.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text(
        "---\ntitle: Memory runbook\ndoc_type: runbook\n---\n# Memory\n\nInspect INFO memory.",
        encoding="utf-8",
    )
    storage = ArtifactStorage(tmp_path / "artifacts")
    redis = FakeRedis()
    pipeline = IngestionPipeline(
        storage,
        settings_config=_settings(),
        index=FakeIndex(redis),
        vectorizer=FakeVectorizer(),
    )

    prepared_count = await pipeline.prepare_source_artifacts(
        source_dir,
        "2026-07-12",
    )

    assert prepared_count == 1
    assert storage.get_batch_manifest("2026-07-12")["total_documents"] == 1
    assert redis.hashes == {}

    results = await pipeline.ingest_prepared_batch("2026-07-12")

    assert results[0]["status"] == "success"
    assert results[0]["batch_date"] == "2026-07-12"
    assert results[0]["documents_processed"] == 1
    assert results[0]["chunks_indexed"] == 1
    ingestion_manifest = storage.current_batch_path / "ingestion_manifest.json"
    assert json.loads(ingestion_manifest.read_text(encoding="utf-8"))["success"] is True


@pytest.mark.asyncio
async def test_ingest_prepared_batch_requires_manifest(tmp_path: Path) -> None:
    storage = ArtifactStorage(tmp_path / "artifacts")
    pipeline = IngestionPipeline(
        storage,
        settings_config=_settings(),
        index=FakeIndex(FakeRedis()),
        vectorizer=FakeVectorizer(),
    )

    with pytest.raises(ValueError, match="manifest"):
        await pipeline.ingest_prepared_batch("2026-07-12")


def test_pipeline_modules_import_without_skills_package() -> None:
    sys.modules.pop("redis_sre_agent.skills", None)
    import redis_sre_agent.pipelines.ingestion.pipeline_workflow_mixin as workflow

    source = inspect.getsource(workflow)
    assert "redis_sre_agent.skills" not in source
    assert "discover_skill_packages" not in source
    assert "_configured_skill_roots" not in source
    assert "redis_sre_agent.skills" not in sys.modules
