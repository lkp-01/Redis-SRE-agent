"""精简 pipeline CLI 的 lazy import、prepare-only 和安全错误测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from click.testing import CliRunner

from redis_sre_agent.cli.main import main
from redis_sre_agent.core.config import Settings
from redis_sre_agent.core.redis import RAGNotReadyError


class FakeStorage:
    instances: list["FakeStorage"] = []

    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path)
        self.current_date = "2026-07-12"
        self.current_batch_path = self.base_path / self.current_date
        self.__class__.instances.append(self)

    def set_batch_date(self, batch_date: str) -> None:
        self.current_date = batch_date
        self.current_batch_path = self.base_path / batch_date


class FakeIngestionPipeline:
    instances: list["FakeIngestionPipeline"] = []
    fail_ingest: bool = False

    def __init__(self, storage: FakeStorage, **_kwargs: Any) -> None:
        self.storage = storage
        self.prepared: list[tuple[Path, str]] = []
        self.ingested: list[str] = []
        self.__class__.instances.append(self)

    async def prepare_source_artifacts(self, source_dir: Path, batch_date: str) -> int:
        self.prepared.append((source_dir, batch_date))
        return 2

    async def ingest_prepared_batch(self, batch_date: str) -> list[dict[str, Any]]:
        self.ingested.append(batch_date)
        if self.fail_ingest:
            raise RAGNotReadyError(
                "redis_search_unavailable",
                "Redis Search/Vector 能力不可用。",
            )
        return [
            {
                "status": "success",
                "batch_date": batch_date,
                "documents_processed": 2,
                "chunks_indexed": 2,
            }
        ]


def _patch_cli(monkeypatch, *, enabled: bool = True) -> Any:
    import redis_sre_agent.cli.pipeline as pipeline_module

    FakeStorage.instances.clear()
    FakeIngestionPipeline.instances.clear()
    FakeIngestionPipeline.fail_ingest = False
    monkeypatch.setattr(pipeline_module, "ArtifactStorage", FakeStorage)
    monkeypatch.setattr(pipeline_module, "IngestionPipeline", FakeIngestionPipeline)
    monkeypatch.setattr(
        pipeline_module,
        "settings",
        Settings(_env_file=None, rag_enabled=enabled),
    )
    return pipeline_module


def test_root_lazy_group_exposes_only_trimmed_pipeline_commands() -> None:
    runner = CliRunner()

    root_help = runner.invoke(main, ["--help"])
    pipeline_help = runner.invoke(main, ["pipeline", "--help"])

    assert root_help.exit_code == 0
    assert "pipeline" in root_help.output
    assert pipeline_help.exit_code == 0
    assert "prepare-sources" in pipeline_help.output
    assert "ingest" in pipeline_help.output
    assert "scrape" not in pipeline_help.output
    assert "cleanup" not in pipeline_help.output


def test_pipeline_disabled_stops_before_artifact_write(monkeypatch, tmp_path: Path) -> None:
    _patch_cli(monkeypatch, enabled=False)
    source_dir = tmp_path / "source_documents"
    source_dir.mkdir()

    result = CliRunner().invoke(
        main,
        [
            "pipeline",
            "prepare-sources",
            "--source-dir",
            str(source_dir),
            "--artifacts-path",
            str(tmp_path / "artifacts"),
            "--prepare-only",
        ],
    )

    assert result.exit_code != 0
    assert "请先启用 RAG" in result.output
    assert FakeStorage.instances == []


def test_prepare_only_writes_artifacts_without_ingesting(monkeypatch, tmp_path: Path) -> None:
    _patch_cli(monkeypatch)
    source_dir = tmp_path / "source_documents"
    source_dir.mkdir()

    result = CliRunner().invoke(
        main,
        [
            "pipeline",
            "prepare-sources",
            "--source-dir",
            str(source_dir),
            "--batch-date",
            "2026-07-12",
            "--artifacts-path",
            str(tmp_path / "artifacts"),
            "--prepare-only",
        ],
    )

    assert result.exit_code == 0, result.output
    pipeline = FakeIngestionPipeline.instances[-1]
    assert pipeline.prepared == [(source_dir, "2026-07-12")]
    assert pipeline.ingested == []
    assert "已准备 2" in result.output


def test_prepare_without_flag_ingests_same_batch(monkeypatch, tmp_path: Path) -> None:
    _patch_cli(monkeypatch)
    source_dir = tmp_path / "source_documents"
    source_dir.mkdir()

    result = CliRunner().invoke(
        main,
        [
            "pipeline",
            "prepare-sources",
            "--source-dir",
            str(source_dir),
            "--batch-date",
            "2026-07-12",
            "--artifacts-path",
            str(tmp_path / "artifacts"),
        ],
    )

    assert result.exit_code == 0, result.output
    pipeline = FakeIngestionPipeline.instances[-1]
    assert pipeline.ingested == ["2026-07-12"]
    assert "已写入 2" in result.output


def test_ingest_reports_safe_not_ready_reason(monkeypatch, tmp_path: Path) -> None:
    _patch_cli(monkeypatch)
    FakeIngestionPipeline.fail_ingest = True

    result = CliRunner().invoke(
        main,
        [
            "pipeline",
            "ingest",
            "--batch-date",
            "2026-07-12",
            "--artifacts-path",
            str(tmp_path / "artifacts"),
        ],
    )

    assert result.exit_code != 0
    assert "redis_search_unavailable" in result.output
    assert "Redis Search/Vector" in result.output
    assert "redis://" not in result.output
