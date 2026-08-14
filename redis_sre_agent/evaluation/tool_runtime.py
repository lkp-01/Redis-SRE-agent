"""把 outcome YAML 中声明的 provider fixture 接到真实 ToolManager。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from redis_sre_agent.core.runtime_overrides import EvalToolDispatchResult, EvalToolRuntime
from redis_sre_agent.evaluation.scenarios import EvalScenario, ToolBehavior


@dataclass(frozen=True)
class ToolTrace:
    provider_family: str
    operation: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class FixtureToolRuntime(EvalToolRuntime):
    scenario: EvalScenario
    traces: list[ToolTrace] = field(default_factory=list)

    @property
    def provider_families(self) -> frozenset[str]:
        """只允许 ToolManager 加载 scenario 显式声明的 provider。"""

        return frozenset(self.scenario.tools)

    async def dispatch_tool_call(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        tool_by_name: Mapping[str, Any],
        routing_table: Mapping[str, Any],
    ) -> EvalToolDispatchResult:
        provider = routing_table.get(tool_name)
        provider_family = str(getattr(provider, "provider_name", ""))
        if provider is None or provider_family not in self.scenario.tools:
            raise RuntimeError(f"离线 outcome 评测拒绝未声明 provider：{tool_name}")
        operation = provider.resolve_operation(tool_name, args)
        provider_tools = self.scenario.tools[provider_family]
        if not operation or operation not in provider_tools:
            raise RuntimeError(f"离线 outcome 评测拒绝未声明操作：{tool_name}")
        result = self._resolve(operation, args, provider_tools[operation])
        self.traces.append(
            ToolTrace(
                provider_family=provider_family,
                operation=operation,
                args=dict(args),
                result=result,
            )
        )
        return EvalToolDispatchResult(result=result)

    def _resolve(self, operation: str, args: dict[str, Any], behavior: ToolBehavior) -> dict[str, Any]:
        reference = behavior.result
        for responder in behavior.responders:
            if _contains(responder.args_contains, args):
                reference = responder.result
                break
        if reference is None:
            raise RuntimeError(f"离线 outcome 评测没有匹配 {operation} 的 fixture")
        payload = json.loads(self.scenario.resolve_fixture_path(reference).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"fixture {reference} 必须是 JSON 对象")
        return payload


def _contains(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    for key, expected_value in expected.items():
        value = actual.get(key)
        if isinstance(expected_value, dict):
            if not isinstance(value, Mapping) or not _contains(expected_value, value):
                return False
        elif value != expected_value:
            return False
    return True


__all__ = ["FixtureToolRuntime", "ToolTrace"]
