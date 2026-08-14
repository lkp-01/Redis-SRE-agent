"""BigKey 确定性断言的最小契约测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from redis_sre_agent.evaluation.assertions import (
    flatten_structured_assertions,
    score_structured_assertions,
)
from redis_sre_agent.evaluation.scenarios import load_eval_scenario
from redis_sre_agent.evaluation.tool_runtime import ToolTrace

_SCENARIO = Path("evals/scenarios/outcome/BigKey/scenario.yaml")


def _successful_trace():
    scenario = load_eval_scenario(_SCENARIO)
    return scenario, [
        ToolTrace(
            provider_family=call.provider_family,
            operation=call.operation,
            args=call.args,
            result={"status": "success"},
        )
        for call in scenario.replay_calls
    ]


def test_structured_assertions_pass_for_replay_transcript() -> None:
    scenario, trace = _successful_trace()
    mechanical_scenario = replace(scenario, required_findings=(), forbidden_claims=())

    results = score_structured_assertions(
        mechanical_scenario,
        tool_trace=trace,
        final_answer=scenario.reference_answer,
    )

    assert results.all_passed
    assert len(results.required_tool_calls) == 3
    assert results.required_response_patterns == []
    assert results.required_findings == []
    assert all(item.passed for item in flatten_structured_assertions(results))


def test_structured_assertions_report_bad_call_and_missing_evidence() -> None:
    scenario, trace = _successful_trace()
    mechanical_scenario = replace(scenario, required_findings=(), forbidden_claims=())
    trace[0] = ToolTrace(
        provider_family="redis_command",
        operation="slowlog",
        args={"count": 1},
        result={"status": "success"},
    )

    results = score_structured_assertions(
        mechanical_scenario,
        tool_trace=trace,
        final_answer="没有足够证据",
    )

    assert not results.all_passed
    assert not results.required_tool_calls[0].passed
    assert results.required_response_patterns == []
