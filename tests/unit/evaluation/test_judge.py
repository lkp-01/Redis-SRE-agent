"""LLM judge 使用 fake 模型验证，严禁触发真实 OpenAI 请求。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langchain_core.messages import AIMessage

import redis_sre_agent.evaluation.judge as judge_module
from redis_sre_agent.evaluation.judge import (
    SREAgentJudge,
    build_default_eval_criteria,
    build_eval_judge_test_case,
    evaluate_eval_scenario_response,
)
from redis_sre_agent.evaluation.scenarios import load_eval_scenario

_SCENARIO = Path("evals/scenarios/outcome/BigKey/scenario.yaml")


def _verbatim_evidence(answer: str) -> str:
    return next(line.strip() for line in answer.splitlines() if line.strip())


def _valid_judge_payload(
    scenario,
    answer: str,
    *,
    score: float = 90,
    required_statuses: dict[str, str] | None = None,
    forbidden_statuses: dict[str, str] | None = None,
) -> dict:
    required_statuses = required_statuses or {}
    forbidden_statuses = forbidden_statuses or {}
    evidence = _verbatim_evidence(answer)
    return {
        "overall_score": score,
        "criteria_scores": {"technical_accuracy": score},
        "strengths": ["grounded"],
        "weaknesses": [],
        "factual_errors": [],
        "required_element_results": [
            {
                "id": f"required_finding_{index}",
                "status": status,
                "evidence": "" if status == "missing" else evidence,
                "explanation": "fixture assessment",
            }
            for index, _finding in enumerate(scenario.required_findings, start=1)
            for status in [required_statuses.get(f"required_finding_{index}", "present")]
        ],
        "forbidden_claim_results": [
            {
                "id": f"forbidden_claim_{index}",
                "status": status,
                "evidence": evidence if status == "violated" else "",
                "explanation": "fixture assessment",
            }
            for index, _claim in enumerate(scenario.forbidden_claims, start=1)
            for status in [forbidden_statuses.get(f"forbidden_claim_{index}", "absent")]
        ],
        "advisory_missing_elements": [],
        "detailed_feedback": "Evidence is consistent.",
    }


class FakeJudgeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = None
        self.bind_kwargs = None

    def bind(self, **kwargs):
        self.bind_kwargs = kwargs
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return AIMessage(content=self.content)


class SequencedFakeJudgeLLM:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.contents.pop(0))


def test_default_judge_uses_deepseek_v4_pro(monkeypatch) -> None:
    fake = FakeJudgeLLM("{}")
    calls = []

    def fake_create_mini_llm(*args, **kwargs):
        calls.append((args, kwargs))
        return fake

    monkeypatch.setattr(judge_module, "create_mini_llm", fake_create_mini_llm)

    judge = SREAgentJudge()

    assert judge.llm is fake
    assert calls == [((), {"model": "deepseek-v4-pro"})]
    assert fake.bind_kwargs == {"response_format": {"type": "json_object"}}


def test_build_judge_case_includes_semantic_expectations() -> None:
    scenario = load_eval_scenario(_SCENARIO)

    payload = build_eval_judge_test_case(scenario, tool_trace=[{"operation": "info"}])

    assert payload["id"] == "outcome/bigkey-latency"
    assert payload["provenance"]["kind"] == "synthetic"
    assert payload["provenance"]["source"] == "local_fixture"
    assert payload["execution_lane"] == "agent_only"
    assert payload["execution_agent"] == "redis_sre"
    assert payload["scope"]["mode"] == "offline"
    assert payload["scope"]["redis_instance"]["id"] == "eval-bigkey"
    assert "connection_url" not in payload["scope"]["redis_instance"]
    assert payload["expectations"]["judge"] == "semantic"
    assert len(payload["expectations"]["required_findings"]) == 5
    assert payload["required_element_catalog"][0] == {
        "id": "required_finding_1",
        "description": scenario.required_findings[0],
    }
    assert payload["forbidden_claim_catalog"][0] == {
        "id": "forbidden_claim_1",
        "description": scenario.forbidden_claims[0],
    }
    assert payload["reference_answer"] == scenario.reference_answer


def test_judge_prompt_treats_only_declared_ids_as_blocking() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    fake = FakeJudgeLLM(json.dumps(_valid_judge_payload(scenario, scenario.reference_answer)))

    asyncio.run(
        evaluate_eval_scenario_response(
            scenario=scenario,
            agent_response=scenario.reference_answer,
            judge=SREAgentJudge(llm=fake),
        )
    )

    prompt = fake.messages[1]["content"]
    assert "required_finding_1" in prompt
    assert "forbidden_claim_1" in prompt
    assert "Reference Answer" not in prompt
    assert "required_element_results" in fake.messages[0]["content"]
    assert "concise grounded" in fake.messages[0]["content"]


def test_judge_parses_fenced_json_with_injected_fake_llm() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    payload = _valid_judge_payload(scenario, scenario.reference_answer, score=92)
    payload["criteria_scores"]["technical_accuracy"] = 95
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    fake = FakeJudgeLLM(f"```json\n{serialized[:-2]},\n}}\n```")

    result = asyncio.run(
        evaluate_eval_scenario_response(
            scenario=scenario,
            agent_response=scenario.reference_answer,
            judge=SREAgentJudge(llm=fake),
        )
    )

    assert result.overall_score == 92
    assert result.criteria_scores["technical_accuracy"] == 95
    assert result.judge_valid is True
    assert len(result.required_element_results) == len(scenario.required_findings)
    assert fake.messages is not None
    assert "824633720" in fake.messages[1]["content"]
    assert "Scenario Provenance" in fake.messages[1]["content"]
    assert "offline-bigkey" in fake.messages[1]["content"]


def test_judge_locally_recovers_unescaped_quotes_inside_string() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    payload = _valid_judge_payload(scenario, scenario.reference_answer, score=82)
    payload["criteria_scores"]["technical_accuracy"] = 80
    payload["weaknesses"] = [
        '一处说"至少 3 次完整的角色切换"，另一处说"主角色切换 3 次"'
    ]
    payload["detailed_feedback"] = "Counts need clearer labels."
    serialized = json.dumps(payload, ensure_ascii=False)
    malformed = serialized.replace(
        r'\"至少 3 次完整的角色切换\"',
        '"至少 3 次完整的角色切换"',
    )
    fake = SequencedFakeJudgeLLM(malformed)

    result = asyncio.run(
        evaluate_eval_scenario_response(
            scenario=scenario,
            agent_response=scenario.reference_answer,
            judge=SREAgentJudge(llm=fake),
        )
    )

    assert result.overall_score == 82
    assert result.weaknesses == [
        '一处说"至少 3 次完整的角色切换"，另一处说"主角色切换 3 次"'
    ]
    assert len(fake.calls) == 1


def test_judge_preserves_smart_quotes_inside_valid_json_strings() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    payload = _valid_judge_payload(scenario, scenario.reference_answer)
    payload["detailed_feedback"] = "failover-timeout 并非简单的“冷却窗口”。"
    fake = FakeJudgeLLM(json.dumps(payload, ensure_ascii=False))

    result = asyncio.run(
        evaluate_eval_scenario_response(
            scenario=scenario,
            agent_response=scenario.reference_answer,
            judge=SREAgentJudge(llm=fake),
        )
    )

    assert result.judge_valid is True
    assert result.detailed_feedback == "failover-timeout 并非简单的“冷却窗口”。"


def test_judge_requests_one_format_repair_for_non_json_payload() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    repaired = json.dumps(
        _valid_judge_payload(scenario, scenario.reference_answer),
        ensure_ascii=False,
    )
    fake = SequencedFakeJudgeLLM("not valid JSON", repaired)

    result = asyncio.run(
        evaluate_eval_scenario_response(
            scenario=scenario,
            agent_response=scenario.reference_answer,
            judge=SREAgentJudge(llm=fake),
        )
    )

    assert result.judge_valid is True
    assert len(fake.calls) == 2
    assert "not valid JSON" in fake.calls[1][1]["content"]


def test_judge_retries_once_when_item_contract_is_invalid() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    invalid = _valid_judge_payload(scenario, scenario.reference_answer)
    invalid["required_element_results"][0]["evidence"] = ""
    repaired = _valid_judge_payload(scenario, scenario.reference_answer)
    fake = SequencedFakeJudgeLLM(
        json.dumps(invalid, ensure_ascii=False),
        json.dumps(repaired, ensure_ascii=False),
    )

    result = asyncio.run(
        evaluate_eval_scenario_response(
            scenario=scenario,
            agent_response=scenario.reference_answer,
            judge=SREAgentJudge(llm=fake),
        )
    )

    assert result.judge_valid is True
    assert result.validation_errors == []
    assert len(fake.calls) == 2
    assert "requires grounded evidence" in fake.calls[1][1]["content"]


def test_judge_confirms_blocking_finding_before_failing_agent() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    first = _valid_judge_payload(
        scenario,
        scenario.reference_answer,
        required_statuses={"required_finding_1": "missing"},
    )
    confirmed = _valid_judge_payload(scenario, scenario.reference_answer)
    fake = SequencedFakeJudgeLLM(
        json.dumps(first, ensure_ascii=False),
        json.dumps(confirmed, ensure_ascii=False),
    )

    result = asyncio.run(
        evaluate_eval_scenario_response(
            scenario=scenario,
            agent_response=scenario.reference_answer,
            judge=SREAgentJudge(llm=fake),
        )
    )

    assert result.judge_valid is True
    assert result.missing_elements == []
    assert len(fake.calls) == 2
    assert "Blocking findings require one independent confirmation" in fake.calls[1][1][
        "content"
    ]


def test_judge_marks_result_invalid_after_one_failed_contract_retry() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    invalid = _valid_judge_payload(scenario, scenario.reference_answer)
    invalid["required_element_results"] = []
    fake = SequencedFakeJudgeLLM(
        json.dumps(invalid, ensure_ascii=False),
        json.dumps(invalid, ensure_ascii=False),
    )

    result = asyncio.run(
        evaluate_eval_scenario_response(
            scenario=scenario,
            agent_response=scenario.reference_answer,
            judge=SREAgentJudge(llm=fake),
        )
    )

    assert result.judge_valid is False
    assert result.validation_errors
    assert "required_element_results" in result.validation_errors[0]
    assert len(fake.calls) == 2


def test_judge_stops_after_one_failed_format_repair() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    fake = SequencedFakeJudgeLLM("not valid: [json", "still not valid: [json")

    result = asyncio.run(
        evaluate_eval_scenario_response(
            scenario=scenario,
            agent_response=scenario.reference_answer,
            judge=SREAgentJudge(llm=fake),
        )
    )

    assert result.overall_score == 0
    assert result.judge_valid is False
    assert result.detailed_feedback.startswith("Evaluation failed due to error:")
    assert "after one format-repair attempt" in result.detailed_feedback
    assert len(fake.calls) == 2


def test_default_rubric_weights_sum_to_one() -> None:
    assert sum(item.weight for item in build_default_eval_criteria()) == 1.0
