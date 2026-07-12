"""本地 Markdown artifact 准备和摄取命令。"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import click

from redis_sre_agent.core.config import settings
from redis_sre_agent.core.redis import RAGNotReadyError
from redis_sre_agent.pipelines.ingestion.processor import IngestionPipeline
from redis_sre_agent.pipelines.scraper.base import ArtifactStorage


def _require_rag_enabled() -> None:
    if not settings.rag_enabled:
        raise click.ClickException("请先启用 RAG（RAG_ENABLED=true，reason=disabled）。")


def _validated_batch_date(batch_date: str | None) -> str | None:
    if batch_date is None:
        return None
    try:
        datetime.strptime(batch_date, "%Y-%m-%d")
    except ValueError as exc:
        raise click.BadParameter("必须使用 YYYY-MM-DD 格式。", param_hint="--batch-date") from exc
    return batch_date


def _not_ready_error(exc: RAGNotReadyError) -> click.ClickException:
    return click.ClickException(f"RAG 未就绪（{exc.reason_code}）：{exc}")


@click.group()
def pipeline() -> None:
    """准备并摄取本地 Markdown knowledge artifacts。"""


@pipeline.command("prepare-sources")
@click.option("--source-dir", "-s", default="source_documents", show_default=True)
@click.option("--batch-date", help="批次日期（YYYY-MM-DD），默认今天。")
@click.option("--prepare-only", is_flag=True, help="只准备 artifact，不写入向量索引。")
@click.option("--artifacts-path", default="./artifacts", show_default=True)
def prepare_sources(
    source_dir: str,
    batch_date: str | None,
    prepare_only: bool,
    artifacts_path: str,
) -> None:
    """把本地 Markdown 转为 batch artifact，并可继续摄取同一批次。"""

    _require_rag_enabled()
    source_path = Path(source_dir)
    if not source_path.exists() or not source_path.is_dir():
        raise click.ClickException("source directory 不存在或不是目录。")
    validated_date = _validated_batch_date(batch_date)
    storage = ArtifactStorage(artifacts_path)
    selected_date = validated_date or storage.current_date
    storage.set_batch_date(selected_date)
    ingestion = IngestionPipeline(storage, settings_config=settings)

    async def _run() -> None:
        try:
            prepared = await ingestion.prepare_source_artifacts(source_path, selected_date)
            click.echo(f"已准备 {prepared} 个本地文档 artifact（batch={selected_date}）。")
            if prepare_only:
                return
            results = await ingestion.ingest_prepared_batch(selected_date)
        except RAGNotReadyError as exc:
            raise _not_ready_error(exc) from exc
        successful = [item for item in results if item.get("status") == "success"]
        if not successful:
            raise click.ClickException("prepared batch 摄取失败。")
        indexed = sum(int(item.get("chunks_indexed") or 0) for item in successful)
        click.echo(f"已写入 {indexed} 个 knowledge chunks。")

    asyncio.run(_run())


@pipeline.command("ingest")
@click.option("--batch-date", required=True, help="要摄取的批次日期（YYYY-MM-DD）。")
@click.option("--artifacts-path", default="./artifacts", show_default=True)
def ingest(batch_date: str, artifacts_path: str) -> None:
    """摄取一个已经准备好的本地 artifact batch。"""

    _require_rag_enabled()
    selected_date = _validated_batch_date(batch_date)
    assert selected_date is not None
    storage = ArtifactStorage(artifacts_path)
    storage.set_batch_date(selected_date)
    ingestion = IngestionPipeline(storage, settings_config=settings)

    async def _run() -> None:
        try:
            results = await ingestion.ingest_prepared_batch(selected_date)
        except RAGNotReadyError as exc:
            raise _not_ready_error(exc) from exc
        successful = [item for item in results if item.get("status") == "success"]
        if not successful:
            raise click.ClickException("prepared batch 摄取失败。")
        processed = sum(int(item.get("documents_processed") or 0) for item in successful)
        indexed = sum(int(item.get("chunks_indexed") or 0) for item in successful)
        click.echo(f"已处理 {processed} 个文档，写入 {indexed} 个 knowledge chunks。")

    asyncio.run(_run())
