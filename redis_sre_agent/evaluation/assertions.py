"""对 outcome replay 的工具轨迹和回答文本执行确定性硬断言。"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from redis_sre_agent.evaluation.report_schema import (
    AssertionStatus,
    EvalAssertionResult,
    StructuredAssertionResult,
    StructuredAssertionResults,
)
from redis_sre_agent.evaluation.scenarios import EvalScenario, ReplayCall
from redis_sre_agent.evaluation.tool_runtime import ToolTrace

_TOKEN_RE = re.compile(r"[\u3400-\u9fff]|[a-z0-9]+(?:[-'][a-z0-9]+)*")
_NEGATION_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("do", "not"),
    ("did", "not"),
    ("does", "not"),
    ("is", "not"),
    ("are", "not"),
    ("was", "not"),
    ("were", "not"),
    ("have", "not"),
    ("has", "not"),
    ("had", "not"),
    ("can", "not"),
    ("cannot",),
    ("can't",),
    ("don't",),
    ("doesn't",),
    ("didn't",),
    ("never",),
    ("no",),
    ("without",),
    ("avoid",),
    ("不",),
    ("没", "有"),
    ("并", "非"),
    ("避", "免"),
)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _tokenize_text(value: Any) -> list[str]:
    return _TOKEN_RE.findall(_normalize_text(value))


def _has_negation_prefix(tokens: Sequence[str], start_index: int) -> bool:
    prefix = tuple(tokens[max(0, start_index - 3) : start_index])
    return any(
        len(prefix) >= len(pattern) and prefix[-len(pattern) :] == pattern
        for pattern in _NEGATION_PREFIXES
    )


def _contains_phrase(text: str, phrase: str) -> bool:
    text_tokens = _tokenize_text(text)
    phrase_tokens = _tokenize_text(phrase)
    if not phrase_tokens or len(text_tokens) < len(phrase_tokens):
        return False
    last_start = len(text_tokens) - len(phrase_tokens) + 1
    for start_index in range(last_start):
        if text_tokens[start_index : start_index + len(phrase_tokens)] != phrase_tokens:
            continue
        if _has_negation_prefix(text_tokens, start_index):
            continue
        return True
    return False


def _matches_response_pattern(text: str | None, pattern: str) -> bool:
    try:
        return re.search(pattern, text or "") is not None
    except re.error:
        return False


def _pass(message: str, *, expected: Any = None, actual: Any = None) -> EvalAssertionResult:
    return EvalAssertionResult(
        status=AssertionStatus.PASSED,
        message=message,
        expected=expected,
        actual=actual,
    )


def _fail(message: str, *, expected: Any = None, actual: Any = None) -> EvalAssertionResult:
    return EvalAssertionResult(
        status=AssertionStatus.FAILED,
        message=message,
        expected=expected,
        actual=actual,
    )


def _normalize_trace(entry: ToolTrace | Mapping[str, Any]) -> ToolTrace:
    if isinstance(entry, ToolTrace):
        return entry
    return ToolTrace(
        provider_family=str(entry.get("provider_family") or "redis_command"),
        operation=str(entry.get("operation", "")),
        args=dict(entry.get("args") or {}),
        result=dict(entry.get("result") or {}),
    )


def _score_required_call(expected: ReplayCall, actual: ToolTrace | None, index: int) -> EvalAssertionResult:
    expected_payload = {
        "provider_family": expected.provider_family,
        "operation": expected.operation,
        "args": expected.args,
    }
    if actual is None:
        return _fail(
            f"缺少第 {index + 1} 个必需工具调用 {expected.operation}",
            expected=expected_payload,
        )
    actual_payload = {
        "provider_family": actual.provider_family,
        "operation": actual.operation,
        "args": actual.args,
    }
    status = str(actual.result.get("status", "")).lower()
    if actual_payload != expected_payload:
        return _fail(
            f"第 {index + 1} 个工具调用与场景 transcript 不匹配",
            expected=expected_payload,
            actual=actual_payload,
        )
    if status != "success":
        return _fail(
            f"工具 {expected.operation} 的 fixture 未返回 success",
            expected="success",
            actual=status or None,
        )
    return _pass(
        f"已观察到必需工具调用 {expected.operation}",
        expected=expected_payload,
        actual=actual_payload,
    )


def score_structured_assertions(
    scenario: EvalScenario,
    *,
    tool_trace: Sequence[ToolTrace | Mapping[str, Any]] | None = None,
    retrieved_sources: Sequence[Any] | None = None,
    final_answer: str | None = None,
    actual_routing_decision: str | None = None,
    tool_identity_map: Sequence[Any] | None = None,
) -> StructuredAssertionResults:
    """复刻 original 的硬断言入口，并收窄到当前 outcome 场景拥有的证据。

    ``retrieved_sources``、``actual_routing_decision`` 和 ``tool_identity_map`` 暂不参与
    outcome 判定，但保留同名参数，后续恢复原版模块时无需改调用方。
    """

    del retrieved_sources, actual_routing_decision, tool_identity_map
    normalized_trace = [_normalize_trace(entry) for entry in (tool_trace or [])]
    answer = final_answer or ""

    required_tool_calls = [
        _score_required_call(
            expected,
            normalized_trace[index] if index < len(normalized_trace) else None,
            index,
        )
        for index, expected in enumerate(scenario.replay_calls)
    ]
    if len(normalized_trace) > len(scenario.replay_calls):
        required_tool_calls.append(
            _fail(
                "观察到场景 transcript 之外的额外工具调用",
                expected=len(scenario.replay_calls),
                actual=len(normalized_trace),
            )
        )

    required_response_patterns = [
        _pass("最终回答命中必需结论模式", expected=pattern, actual=answer)
        if _matches_response_pattern(answer, pattern)
        else _fail("最终回答缺少必需结论模式", expected=pattern, actual=answer)
        for pattern in scenario.required_response_patterns
    ]
    required_findings = [
        _pass(f"最终回答包含必需结论：{finding}", expected=finding, actual=answer)
        if _contains_phrase(answer, finding)
        else _fail(f"最终回答缺少必需结论：{finding}", expected=finding, actual=answer)
        for finding in scenario.required_findings
    ]
    forbidden_claims = [
        _fail(f"最终回答包含禁止结论：{claim}", expected=claim, actual=answer)
        if _contains_phrase(answer, claim)
        else _pass(f"最终回答未包含禁止结论：{claim}", expected=claim)
        for claim in scenario.forbidden_claims
    ]

    return StructuredAssertionResults(
        required_tool_calls=required_tool_calls,
        required_response_patterns=required_response_patterns,
        forbidden_claims=forbidden_claims,
        required_findings=required_findings,
    )


def flatten_structured_assertions(
    results: StructuredAssertionResults,
) -> list[StructuredAssertionResult]:
    """把分组结果转换为与 original 报告模块兼容的扁平行。"""

    rows: list[StructuredAssertionResult] = []
    grouped = {
        "required_tool_call": results.required_tool_calls,
        "forbidden_tool_call": results.forbidden_tool_calls,
        "required_source": results.required_sources,
        "required_response_pattern": results.required_response_patterns,
        "forbidden_claim": results.forbidden_claims,
        "required_finding": results.required_findings,
    }
    for assertion_type, entries in grouped.items():
        for entry in entries:
            rows.append(
                StructuredAssertionResult(
                    assertion_type=assertion_type,
                    passed=entry.passed,
                    details=entry.message,
                    expected=entry.expected,
                    observed=entry.actual,
                )
            )
    return rows


__all__ = ["flatten_structured_assertions", "score_structured_assertions"]
