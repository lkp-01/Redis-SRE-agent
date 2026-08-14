"""唯一 BigKey evaluation 的端到端离线测试。"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from langchain_core.messages import AIMessage, ToolMessage

from redis_sre_agent.cli.main import main
from redis_sre_agent.cli.eval import _echo_console_safe
from redis_sre_agent.evaluation.judge import SREAgentJudge
import redis_sre_agent.evaluation.runtime as runtime_module
from redis_sre_agent.evaluation.runtime import run_bigkey_scenario
from redis_sre_agent.evaluation.scenarios import load_eval_scenario
from redis_sre_agent.tools.diagnostics.redis_command.provider import RedisCommandToolProvider


_SCENARIO = Path("evals/scenarios/outcome/BigKey/scenario.yaml")


class FakeJudgeLLM:
    def __init__(
        self,
        score: float,
        *,
        factual_errors: list[str] | None = None,
        missing_elements: list[str] | None = None,
        partial_elements: list[str] | None = None,
        violated_forbidden_claims: list[str] | None = None,
        advisory_missing_elements: list[str] | None = None,
    ) -> None:
        self.score = score
        self.factual_errors = factual_errors or []
        self.missing_elements = missing_elements or []
        self.partial_elements = partial_elements or []
        self.violated_forbidden_claims = violated_forbidden_claims or []
        self.advisory_missing_elements = advisory_missing_elements or []
        self.calls = 0

    async def ainvoke(self, messages) -> AIMessage:
        self.calls += 1
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
                    "overall_score": self.score,
                    "criteria_scores": {"technical_accuracy": self.score},
                    "strengths": [],
                    "weaknesses": [],
                    "factual_errors": self.factual_errors,
                    "required_element_results": [
                        {
                            "id": item_id,
                            "status": (
                                "missing"
                                if item_id in self.missing_elements
                                else "partial"
                                if item_id in self.partial_elements
                                else "present"
                            ),
                            "evidence": ""
                            if item_id in self.missing_elements
                            else evidence,
                            "explanation": "fake assessment",
                        }
                        for item_id in required_ids
                    ],
                    "forbidden_claim_results": [
                        {
                            "id": item_id,
                            "status": "violated"
                            if item_id in self.violated_forbidden_claims
                            else "absent",
                            "evidence": evidence
                            if item_id in self.violated_forbidden_claims
                            else "",
                            "explanation": "fake assessment",
                        }
                        for item_id in forbidden_ids
                    ],
                    "advisory_missing_elements": self.advisory_missing_elements,
                    "detailed_feedback": "fake judge result",
                }
            )
        )


def _fake_judge(
    score: float,
    *,
    factual_errors: list[str] | None = None,
    missing_elements: list[str] | None = None,
    partial_elements: list[str] | None = None,
    violated_forbidden_claims: list[str] | None = None,
    advisory_missing_elements: list[str] | None = None,
):
    llm = FakeJudgeLLM(
        score,
        factual_errors=factual_errors,
        missing_elements=missing_elements,
        partial_elements=partial_elements,
        violated_forbidden_claims=violated_forbidden_claims,
        advisory_missing_elements=advisory_missing_elements,
    )
    return SREAgentJudge(llm=llm), llm


def test_bigkey_yaml_loads_provenance_scope_and_execution_agent() -> None:
    scenario = load_eval_scenario(_SCENARIO)

    assert scenario.provenance.kind == "synthetic"
    assert scenario.provenance.source == "local_fixture"
    assert scenario.provenance.owner == "redis-sre-eval"
    assert scenario.provenance.reviewed_at == "2026-08-14"
    assert scenario.execution_agent == "redis_sre"
    assert scenario.scope.mode == "offline"
    assert scenario.scope.redis_instance.id == "eval-bigkey"
    assert scenario.scope.redis_instance.name == "offline-bigkey"
    assert scenario.scope.redis_instance.connection_url == "redis://evaluation.invalid:6379/0"


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "knowledge", {"enabled": False}),
        ("execution", "agnt", "redis_sre"),
    ],
)
def test_unknown_scenario_fields_are_rejected(
    monkeypatch,
    section: str | None,
    field: str,
    value,
) -> None:
    payload = yaml.safe_load(_SCENARIO.read_text(encoding="utf-8"))
    target = payload if section is None else payload[section]
    target[field] = value
    monkeypatch.setattr("redis_sre_agent.evaluation.scenarios.yaml.safe_load", lambda _: payload)

    with pytest.raises(ValueError, match=field):
        load_eval_scenario(_SCENARIO)


def test_runtime_instance_is_built_from_scenario_scope() -> None:
    scenario = load_eval_scenario(_SCENARIO)

    instance = runtime_module._eval_instance(scenario.scope)

    assert instance.id == scenario.scope.redis_instance.id
    assert instance.name == scenario.scope.redis_instance.name
    assert instance.connection_url.get_secret_value() == scenario.scope.redis_instance.connection_url
    assert instance.environment == scenario.scope.redis_instance.environment
    assert instance.usage == scenario.scope.redis_instance.usage


def test_runtime_rejects_an_unsupported_execution_agent() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    unsupported = replace(scenario, execution_agent="unknown_agent")

    with pytest.raises(ValueError, match="execution.agent"):
        runtime_module._create_eval_agent(unsupported, FakeAnswerLLM("unused"))


class FakeAnswerLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0
        self.tool_messages: list[ToolMessage] = []

    async def ainvoke(self, messages) -> AIMessage:
        self.calls += 1
        self.tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        return AIMessage(content=self.answer)


def test_bigkey_yaml_runs_through_real_agent_without_network(monkeypatch) -> None:
    scenario = load_eval_scenario(_SCENARIO)
    judge, judge_llm = _fake_judge(92)
    generated_answer = "这是被测模型根据三条工具证据现场生成的 BigKey 诊断。"
    answer_llm = FakeAnswerLLM(generated_answer)

    def network_forbidden(self):
        raise AssertionError("fixture runtime must prevent real Redis access")

    monkeypatch.setattr(RedisCommandToolProvider, "get_client", network_forbidden)
    result = asyncio.run(
        run_bigkey_scenario(scenario, answer_llm=answer_llm, judge=judge)
    )

    assert result.passed, result.assertion_errors
    assert result.judge_passed is True
    assert result.judge_result is not None
    assert result.judge_result.overall_score == 92
    assert judge_llm.calls == 1
    assert answer_llm.calls == 1
    assert len(answer_llm.tool_messages) == 3
    assert any("user:timeline:10086" in str(message.content) for message in answer_llm.tool_messages)
    assert [item["provider_family"] for item in result.tool_trace] == [
        "redis_command",
        "redis_command",
        "redis_command",
    ]
    assert [(item["operation"], item["args"]) for item in result.tool_trace] == [
        ("info", {"section": "memory"}),
        ("slowlog", {"count": 5}),
        (
            "bigkey_scan",
            {
                "threshold_bytes": 1048576,
                "max_keys": 20000,
                "scan_count": 500,
                "top_n": 20,
                "time_limit_ms": 5000,
            },
        ),
    ]
    assert result.final_answer == generated_answer
    assert result.final_answer != scenario.reference_answer


def test_fixture_path_cannot_escape_scenario_directory() -> None:
    scenario = load_eval_scenario(_SCENARIO)

    with pytest.raises(ValueError, match="不能越出"):
        scenario.resolve_fixture_path("../../outside.json")


def test_unknown_judge_mode_is_rejected(monkeypatch) -> None:
    payload = yaml.safe_load(_SCENARIO.read_text(encoding="utf-8"))
    payload["expectations"]["judge"] = "typo"
    monkeypatch.setattr("redis_sre_agent.evaluation.scenarios.yaml.safe_load", lambda _: payload)

    with pytest.raises(ValueError, match="expectations.judge"):
        load_eval_scenario(_SCENARIO)


def test_missing_required_finding_fails_the_result() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    judge, judge_llm = _fake_judge(90, missing_elements=["required_finding_1"])
    answer_llm = FakeAnswerLLM("没有足够证据")
    result = asyncio.run(
        run_bigkey_scenario(
            scenario,
            answer_llm=answer_llm,
            judge=judge,
        )
    )

    assert not result.passed
    assert result.structured_assertions.all_passed
    assert result.judge_passed is False
    assert result.judge_result is not None
    assert result.judge_result.missing_elements == [scenario.required_findings[0]]
    assert judge_llm.calls == 2


def test_reference_only_omission_is_non_blocking_advice() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    optional_omission = "没有复述参考答案中的可选调优细节"
    judge, _ = _fake_judge(88, advisory_missing_elements=[optional_omission])

    result = asyncio.run(
        run_bigkey_scenario(
            scenario,
            answer_llm=FakeAnswerLLM("覆盖全部声明必需项的诊断"),
            judge=judge,
        )
    )

    assert result.passed is True
    assert result.judge_passed is True
    assert result.judge_result is not None
    assert result.judge_result.missing_elements == []
    assert result.judge_result.advisory_missing_elements == [optional_omission]


def test_partial_required_finding_is_reported_but_does_not_hard_fail() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    judge, _ = _fake_judge(85, partial_elements=["required_finding_4"])

    result = asyncio.run(
        run_bigkey_scenario(
            scenario,
            answer_llm=FakeAnswerLLM("部分覆盖必需项的高质量诊断"),
            judge=judge,
        )
    )

    assert result.passed is True
    assert result.judge_passed is True
    assert result.judge_result is not None
    assert result.judge_result.partial_elements == [scenario.required_findings[3]]
    assert result.judge_result.missing_elements == []


def test_forbidden_claim_violation_fails_even_when_score_is_high() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    judge, _ = _fake_judge(
        95,
        violated_forbidden_claims=["forbidden_claim_1"],
    )

    result = asyncio.run(
        run_bigkey_scenario(
            scenario,
            answer_llm=FakeAnswerLLM("包含禁止结论的诊断"),
            judge=judge,
        )
    )

    assert result.judge_result is not None
    assert result.judge_result.violated_forbidden_claims == [
        scenario.forbidden_claims[0]
    ]
    assert result.judge_passed is False
    assert result.passed is False


def test_factual_error_fails_even_when_judge_score_is_high() -> None:
    scenario = load_eval_scenario(_SCENARIO)
    judge, _ = _fake_judge(95, factual_errors=["把无关命令错误归因为 BigKey"])

    result = asyncio.run(
        run_bigkey_scenario(
            scenario,
            answer_llm=FakeAnswerLLM("包含事实错误的诊断"),
            judge=judge,
        )
    )

    assert result.judge_result is not None
    assert result.judge_result.overall_score == 95
    assert result.judge_result.factual_errors
    assert result.judge_passed is False
    assert result.passed is False


def test_eval_cli_prints_json_and_succeeds(monkeypatch) -> None:
    judge, judge_llm = _fake_judge(90)
    answer_llm = FakeAnswerLLM("CLI 现场生成的诊断")
    monkeypatch.setattr("redis_sre_agent.evaluation.runtime.SREAgentJudge", lambda: judge)
    monkeypatch.setattr("redis_sre_agent.evaluation.runtime.create_llm", lambda: answer_llm)

    result = CliRunner().invoke(main, ["eval", "run", str(_SCENARIO), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["scenario_id"] == "outcome/bigkey-latency"
    assert payload["passed"] is True
    assert payload["judge_passed"] is True
    assert payload["judge_result"]["overall_score"] == 90
    assert judge_llm.calls == 1
    assert answer_llm.calls == 1


def test_eval_cli_plain_output_includes_judge_summary(monkeypatch) -> None:
    judge, judge_llm = _fake_judge(90)
    answer_llm = FakeAnswerLLM("CLI 现场生成的诊断")
    monkeypatch.setattr("redis_sre_agent.evaluation.runtime.SREAgentJudge", lambda: judge)
    monkeypatch.setattr("redis_sre_agent.evaluation.runtime.create_llm", lambda: answer_llm)

    result = CliRunner().invoke(main, ["eval", "run", str(_SCENARIO)])

    assert result.exit_code == 0, result.output
    assert "scenario_id=outcome/bigkey-latency" in result.output
    assert "passed=true" in result.output
    assert "judge_score=90.0" in result.output
    assert "judge_passed=true" in result.output
    assert "factual_errors=0" in result.output
    assert "missing_elements=0" in result.output
    assert "partial_elements=0" in result.output
    assert "violated_forbidden_claims=0" in result.output
    assert "advisory_missing_elements=0" in result.output
    assert "judge_feedback=fake judge result" in result.output
    assert "CLI 现场生成的诊断" in result.output
    assert judge_llm.calls == 1
    assert answer_llm.calls == 1


def test_eval_cli_safely_prints_non_gbk_agent_text(monkeypatch) -> None:
    judge, _ = _fake_judge(90)
    answer_llm = FakeAnswerLLM("⚠ Redis 诊断通过")
    monkeypatch.setattr("redis_sre_agent.evaluation.runtime.SREAgentJudge", lambda: judge)
    monkeypatch.setattr("redis_sre_agent.evaluation.runtime.create_llm", lambda: answer_llm)

    result = CliRunner().invoke(main, ["eval", "run", str(_SCENARIO)])

    assert result.exit_code == 0, result.output
    assert "Redis 诊断通过" in result.output


def test_console_safe_echo_escapes_characters_missing_from_gbk(monkeypatch) -> None:
    class GbkStream:
        encoding = "gbk"

    captured: list[str] = []
    monkeypatch.setattr("redis_sre_agent.cli.eval.sys.stdout", GbkStream())
    monkeypatch.setattr("redis_sre_agent.cli.eval.click.echo", captured.append)

    _echo_console_safe("⚠ Redis")

    assert captured == [r"\u26a0 Redis"]


def test_eval_cli_plain_output_lists_judge_failures(monkeypatch) -> None:
    judge, _ = _fake_judge(
        95,
        factual_errors=["引用了工具结果中不存在的实例名"],
        advisory_missing_elements=["没有明确表达剩余不确定性"],
    )
    monkeypatch.setattr("redis_sre_agent.evaluation.runtime.SREAgentJudge", lambda: judge)
    monkeypatch.setattr(
        "redis_sre_agent.evaluation.runtime.create_llm",
        lambda: FakeAnswerLLM("存在证据问题的诊断"),
    )

    result = CliRunner().invoke(main, ["eval", "run", str(_SCENARIO)])
    output_lines = result.output.splitlines()

    assert result.exit_code != 0
    assert "factual_errors=1" in output_lines
    assert "factual_error[1]=引用了工具结果中不存在的实例名" in result.output
    assert "missing_elements=0" in output_lines
    assert "advisory_missing_elements=1" in output_lines
    assert "advisory_missing_element[1]=没有明确表达剩余不确定性" in result.output


class InvalidContractJudgeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, _messages) -> AIMessage:
        self.calls += 1
        return AIMessage(
            content=json.dumps(
                {
                    "overall_score": 90,
                    "criteria_scores": {},
                    "strengths": [],
                    "weaknesses": [],
                    "factual_errors": [],
                    "required_element_results": [],
                    "forbidden_claim_results": [],
                    "advisory_missing_elements": [],
                    "detailed_feedback": "invalid contract",
                }
            )
        )


def test_invalid_judge_contract_is_not_reported_as_agent_failure(monkeypatch) -> None:
    judge_llm = InvalidContractJudgeLLM()
    judge = SREAgentJudge(llm=judge_llm)
    monkeypatch.setattr("redis_sre_agent.evaluation.runtime.SREAgentJudge", lambda: judge)
    monkeypatch.setattr(
        "redis_sre_agent.evaluation.runtime.create_llm",
        lambda: FakeAnswerLLM("结构正确但 Judge 无法评估的回答"),
    )

    result = CliRunner().invoke(main, ["eval", "run", str(_SCENARIO)])
    output_lines = result.output.splitlines()

    assert result.exit_code != 0
    assert "passed=invalid" in output_lines
    assert "judge_valid=false" in output_lines
    assert "Error: Outcome evaluation invalid" in result.output
    assert "Error: Outcome evaluation failed" not in result.output
    assert judge_llm.calls == 2
