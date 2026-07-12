"""knowledge search CLI 与 shared helper 输出测试。"""

from __future__ import annotations

from typing import Any

from click.testing import CliRunner

from redis_sre_agent.cli.main import main


def _result() -> dict[str, Any]:
    return {
        "status": "success",
        "reason_code": "ready",
        "retrieval_kind": "knowledge_search",
        "retrieval_label": "Knowledge search",
        "results_count": 1,
        "results": [
            {
                "title": "Latency runbook",
                "source": "file://shared/latency.md",
                "document_hash": "doc-hash",
                "chunk_index": 2,
                "score": 0.125,
                "content": "Inspect SLOWLOG before changing configuration.",
            }
        ],
    }


def test_root_exposes_trimmed_knowledge_search_only() -> None:
    runner = CliRunner()
    root = runner.invoke(main, ["--help"])
    help_result = runner.invoke(main, ["knowledge", "--help"])

    assert root.exit_code == 0
    assert "knowledge" in root.output
    assert help_result.exit_code == 0
    assert "search" in help_result.output
    assert "ingest" not in help_result.output
    assert "fragments" not in help_result.output


def test_knowledge_search_outputs_source_identity_and_score(monkeypatch) -> None:
    import redis_sre_agent.cli.knowledge as knowledge_module

    captured: dict[str, Any] = {}

    async def fake_helper(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _result()

    monkeypatch.setattr(knowledge_module, "search_knowledge_base_helper", fake_helper)
    result = CliRunner().invoke(
        main,
        [
            "knowledge",
            "search",
            "redis",
            "latency",
            "--limit",
            "3",
            "--offset",
            "1",
            "--version",
            "latest",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Latency runbook" in result.output
    assert "file://shared/latency.md" in result.output
    assert "doc-hash" in result.output
    assert "chunk=2" in result.output
    assert "score=0.125" in result.output
    assert "Inspect SLOWLOG" in result.output
    assert captured["query"] == "redis latency"
    assert captured["limit"] == 3
    assert captured["offset"] == 1


def test_knowledge_search_unavailable_is_not_reported_as_empty_success(monkeypatch) -> None:
    import redis_sre_agent.cli.knowledge as knowledge_module

    async def unavailable(**_kwargs: Any) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "reason_code": "index_missing",
            "message": "knowledge index 尚未创建。",
            "results": [],
        }

    monkeypatch.setattr(knowledge_module, "search_knowledge_base_helper", unavailable)
    result = CliRunner().invoke(main, ["knowledge", "search", "memory"])

    assert result.exit_code != 0
    assert "index_missing" in result.output
    assert "尚未创建" in result.output
    assert "No results" not in result.output
