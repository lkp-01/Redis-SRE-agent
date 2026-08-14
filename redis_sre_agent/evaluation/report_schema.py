"""evaluation 的最小结构化断言结果模型。

original 的报告模块还包含 live suite、baseline 和落盘产物；当前 outcome 切片只保留
``assertions.py`` 真正需要的结果类型，后续可以在不改变这些公共字段的前提下扩展。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AssertionStatus(str, Enum):
    """单条确定性断言的结果。"""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EvalAssertionResult(BaseModel):
    """保留预期值和实际值，便于 CLI 或后续报告模块解释失败原因。"""

    model_config = ConfigDict(extra="forbid")

    status: AssertionStatus
    message: str | None = None
    expected: Any = None
    actual: Any = None

    @property
    def passed(self) -> bool:
        return self.status == AssertionStatus.PASSED


class StructuredAssertionResult(BaseModel):
    """扁平化后的断言行，给后续 JSON 报告扩展使用。"""

    model_config = ConfigDict(extra="forbid")

    assertion_type: str
    passed: bool
    details: str | None = None
    expected: Any = None
    observed: Any = None


class StructuredAssertionResults(BaseModel):
    """当前 outcome 切片支持的分组断言结果。"""

    model_config = ConfigDict(extra="forbid")

    required_tool_calls: list[EvalAssertionResult] = Field(default_factory=list)
    forbidden_tool_calls: list[EvalAssertionResult] = Field(default_factory=list)
    required_sources: list[EvalAssertionResult] = Field(default_factory=list)
    required_response_patterns: list[EvalAssertionResult] = Field(default_factory=list)
    forbidden_claims: list[EvalAssertionResult] = Field(default_factory=list)
    required_findings: list[EvalAssertionResult] = Field(default_factory=list)
    expected_routing_decision: EvalAssertionResult | None = None
    all_passed: bool | None = None

    @model_validator(mode="after")
    def _derive_all_passed(self) -> "StructuredAssertionResults":
        outcomes = [
            *self.required_tool_calls,
            *self.forbidden_tool_calls,
            *self.required_sources,
            *self.required_response_patterns,
            *self.forbidden_claims,
            *self.required_findings,
        ]
        if self.expected_routing_decision is not None:
            outcomes.append(self.expected_routing_decision)
        if self.all_passed is None:
            self.all_passed = all(item.status != AssertionStatus.FAILED for item in outcomes)
        return self


__all__ = [
    "AssertionStatus",
    "EvalAssertionResult",
    "StructuredAssertionResult",
    "StructuredAssertionResults",
]
