"""阶段三 CLI query 入口测试。"""

from __future__ import annotations

import json

from redis_sre_agent.cli.main import main
from redis_sre_agent.core import targets as targets_module


def test_cli_status_reports_stage_three(capsys) -> None:
    assert main(["status"]) == 0

    captured = capsys.readouterr()
    assert "阶段三" in captured.out


def test_cli_query_prints_agent_response(monkeypatch, capsys) -> None:
    async def fake_get_instances():
        return []

    async def fake_get_clusters():
        return []

    monkeypatch.setattr(targets_module, "get_instances", fake_get_instances)
    monkeypatch.setattr(targets_module, "get_clusters", fake_get_clusters)

    assert main(["query", "check", "redis", "--target", "prod-cache", "--user-id", "u1"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert "阶段三曳光弹未绑定 Redis target" in payload["response"]
    assert payload["search_results"] == []
    assert payload["tool_envelopes"][0]["name"] == "resolve_redis_targets"
    assert payload["tool_envelopes"][0]["status"] == "no_match"
