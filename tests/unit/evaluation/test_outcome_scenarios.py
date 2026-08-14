"""新增 outcome 场景的库存、fixture 和真实 Agent 离线回放测试。"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from redis_sre_agent.core.config import Settings
from redis_sre_agent.core.runtime_overrides import eval_tool_runtime_scope
from redis_sre_agent.evaluation.judge import SREAgentJudge
from redis_sre_agent.evaluation.runtime import _eval_instance, run_eval_scenario
from redis_sre_agent.evaluation.scenarios import load_eval_scenario
from redis_sre_agent.evaluation.tool_runtime import FixtureToolRuntime
from redis_sre_agent.tools.diagnostics.redis_command.provider import RedisCommandToolProvider
from redis_sre_agent.tools.logs.loki.provider import LokiToolProvider
from redis_sre_agent.tools.metrics.prometheus.provider import PrometheusToolProvider
import redis_sre_agent.tools.manager as manager_module
from redis_sre_agent.tools.manager import ToolManager


_OUTCOME_ROOT = Path("evals/scenarios/outcome")
_NEW_SCENARIO_NAMES = (
    "ConnectionReadyStorm",
    "AOFEventLoopStall",
    "SlowClientOutputBuffer",
    "ForkMemorySpike",
    "DiskFullFailures",
    "SwapThrashing",
    "NetworkDegradation",
    "CPUThreadContention",
    "SlowStorageIO",
    "FailoverFlapping",
    "ReplicaLagOutputBuffer",
)
_NEW_SCENARIOS = tuple(_OUTCOME_ROOT / name / "scenario.yaml" for name in _NEW_SCENARIO_NAMES)
_EXPECTED_IDS = {
    "outcome/connection-ready-storm",
    "outcome/aof-event-loop-stall",
    "outcome/slow-client-output-buffer",
    "outcome/fork-memory-spike",
    "outcome/disk-full-failures",
    "outcome/swap-thrashing",
    "outcome/network-degradation",
    "outcome/cpu-thread-contention",
    "outcome/slow-storage-io",
    "outcome/failover-flapping",
    "outcome/replica-lag-output-buffer",
}


class FakeAnswerLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.tool_messages: list[ToolMessage] = []

    async def ainvoke(self, messages) -> AIMessage:
        self.tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        return AIMessage(content=self.answer)


class FakeJudgeLLM:
    async def ainvoke(self, messages) -> AIMessage:
        prompt = messages[1]["content"]
        required_ids = sorted(set(re.findall(r'"id": "(required_finding_\d+)"', prompt)))
        forbidden_ids = sorted(set(re.findall(r'"id": "(forbidden_claim_\d+)"', prompt)))
        answer = prompt.split("## Agent Response to Evaluate\n", 1)[1].split(
            "\n\nPlease evaluate",
            1,
        )[0].strip()
        evidence = next(line.strip() for line in answer.splitlines() if line.strip())
        return AIMessage(
            content=json.dumps(
                {
                    "overall_score": 95,
                    "criteria_scores": {"technical_accuracy": 95},
                    "strengths": ["grounded in replay fixtures"],
                    "weaknesses": [],
                    "factual_errors": [],
                    "required_element_results": [
                        {
                            "id": item_id,
                            "status": "present",
                            "evidence": evidence,
                            "explanation": "fixture assessment",
                        }
                        for item_id in required_ids
                    ],
                    "forbidden_claim_results": [
                        {
                            "id": item_id,
                            "status": "absent",
                            "evidence": "",
                            "explanation": "fixture assessment",
                        }
                        for item_id in forbidden_ids
                    ],
                    "advisory_missing_elements": [],
                    "detailed_feedback": "offline outcome fixture passed",
                }
            )
        )


def test_outcome_inventory_contains_bigkey_plus_eleven_new_scenarios() -> None:
    discovered = set(_OUTCOME_ROOT.glob("*/scenario.yaml"))
    expected = {_OUTCOME_ROOT / "BigKey" / "scenario.yaml", *_NEW_SCENARIOS}

    assert discovered == expected


def test_new_outcome_scenarios_have_unique_ids_and_local_success_fixtures() -> None:
    scenarios = [load_eval_scenario(path) for path in _NEW_SCENARIOS]

    assert {scenario.id for scenario in scenarios} == _EXPECTED_IDS
    assert all(scenario.provenance.kind == "synthetic" for scenario in scenarios)
    assert all(scenario.scope.mode == "offline" for scenario in scenarios)
    assert all(len(scenario.replay_calls) == 3 for scenario in scenarios)
    for scenario in scenarios:
        assert scenario.max_tool_steps >= len(scenario.replay_calls)
        for call in scenario.replay_calls:
            behavior = scenario.tools[call.provider_family][call.operation]
            references = [behavior.result, *(item.result for item in behavior.responders)]
            for reference in (item for item in references if item is not None):
                payload = json.loads(
                    scenario.resolve_fixture_path(reference).read_text(encoding="utf-8")
                )
                assert payload["status"] == "success"


def test_failover_flapping_metrics_align_with_sentinel_timeline() -> None:
    scenario_root = _OUTCOME_ROOT / "FailoverFlapping"
    prometheus = json.loads(
        (scenario_root / "fixtures/tools/prometheus-role-changes.json").read_text(
            encoding="utf-8"
        )
    )
    loki = json.loads(
        (scenario_root / "fixtures/tools/loki-sentinel-failovers.json").read_text(
            encoding="utf-8"
        )
    )

    redis_up = next(
        series
        for series in prometheus["data"]
        if series["metric"].get("__name__") == "redis_up"
    )
    outage_timestamps = {
        int(timestamp) for timestamp, value in redis_up["values"] if value == "0"
    }
    switch_timestamps = {
        int(timestamp_ns) // 1_000_000_000
        for stream in loki["data"]["result"]
        for timestamp_ns, line in stream["values"]
        if "+switch-master" in line
    }
    all_metric_timestamps = {
        int(timestamp)
        for series in prometheus["data"]
        for timestamp, _value in series["values"]
    }

    assert outage_timestamps == switch_timestamps
    assert min(all_metric_timestamps) >= min(switch_timestamps) - 30 * 60
    assert max(all_metric_timestamps) <= max(switch_timestamps) + 30


@pytest.mark.asyncio
async def test_eval_tool_manager_loads_only_scenario_declared_providers(monkeypatch) -> None:
    monkeypatch.setattr(
        manager_module,
        "settings",
        Settings(
            _env_file=None,
            rag_enabled=False,
            mcp_servers={},
            tool_providers=[
                "redis_sre_agent.tools.diagnostics.redis_command.provider.RedisCommandToolProvider"
            ],
        ),
    )
    cases = [
        (_OUTCOME_ROOT / "BigKey" / "scenario.yaml", {"redis_command"}),
        (
            _OUTCOME_ROOT / "NetworkDegradation" / "scenario.yaml",
            {"redis_command", "prometheus", "loki"},
        ),
    ]

    for path, expected_families in cases:
        scenario = load_eval_scenario(path)
        fixture_runtime = FixtureToolRuntime(scenario)
        with eval_tool_runtime_scope(fixture_runtime):
            async with ToolManager(redis_instance=_eval_instance(scenario.scope)) as manager:
                loaded_families = {
                    family
                    for family in ("redis_command", "prometheus", "loki")
                    if manager.get_tools_by_provider_names([family])
                }

        assert loaded_families == expected_families


@pytest.mark.parametrize("scenario_path", _NEW_SCENARIOS, ids=_NEW_SCENARIO_NAMES)
def test_new_outcome_scenario_runs_offline_through_real_agent(
    monkeypatch,
    scenario_path: Path,
) -> None:
    scenario = load_eval_scenario(scenario_path)
    answer_llm = FakeAnswerLLM(scenario.reference_answer)

    def network_forbidden(*_args, **_kwargs):
        raise AssertionError("outcome fixture runtime must prevent external access")

    async def async_network_forbidden(*_args, **_kwargs):
        raise AssertionError("outcome fixture runtime must prevent external access")

    monkeypatch.setattr(RedisCommandToolProvider, "get_client", network_forbidden)
    monkeypatch.setattr(PrometheusToolProvider, "get_client", network_forbidden)
    monkeypatch.setattr(LokiToolProvider, "_request", async_network_forbidden)

    result = asyncio.run(
        run_eval_scenario(
            scenario,
            answer_llm=answer_llm,
            judge=SREAgentJudge(llm=FakeJudgeLLM()),
        )
    )

    assert result.passed, result.assertion_errors
    assert result.judge_passed is True
    assert len(answer_llm.tool_messages) == len(scenario.replay_calls)
    assert [
        (item["provider_family"], item["operation"], item["args"])
        for item in result.tool_trace
    ] == [
        (call.provider_family, call.operation, call.args)
        for call in scenario.replay_calls
    ]
