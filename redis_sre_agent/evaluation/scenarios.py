"""Outcome 离线 replay 场景的 YAML 合同。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_SUPPORTED_PROVIDER_FAMILIES = {"redis_command", "prometheus", "loki"}


@dataclass(frozen=True)
class ToolResponder:
    args_contains: dict[str, Any]
    result: str


@dataclass(frozen=True)
class ToolBehavior:
    result: str | None = None
    responders: tuple[ToolResponder, ...] = ()


@dataclass(frozen=True)
class ReplayCall:
    provider_family: str
    operation: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioProvenance:
    kind: str
    source: str
    owner: str
    reviewed_at: str


@dataclass(frozen=True)
class ScenarioRedisInstance:
    id: str
    name: str
    connection_url: str
    environment: str
    usage: str
    description: str
    status: str
    created_by: str


@dataclass(frozen=True)
class ScenarioScope:
    mode: str
    redis_instance: ScenarioRedisInstance


@dataclass(frozen=True)
class EvalScenario:
    id: str
    name: str
    description: str
    provenance: ScenarioProvenance
    execution_lane: str
    execution_agent: str
    scope: ScenarioScope
    query: str
    max_tool_steps: int
    llm_mode: str
    judge: str
    replay_calls: tuple[ReplayCall, ...]
    reference_answer: str
    tools: dict[str, dict[str, ToolBehavior]]
    required_response_patterns: tuple[str, ...]
    required_findings: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    _source_path: Path

    @property
    def source_path(self) -> Path:
        return self._source_path

    def resolve_fixture_path(self, reference: str) -> Path:
        """只接受场景目录内的相对 fixture，阻止读取任意本地文件。"""

        reference_path = Path(reference)
        if reference_path.is_absolute():
            raise ValueError("评测 fixture 必须使用相对路径")
        scenario_root = self._source_path.parent.resolve()
        candidate = (scenario_root / reference_path).resolve()
        try:
            candidate.relative_to(scenario_root)
        except ValueError as exc:
            raise ValueError("评测 fixture 不能越出场景目录") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"评测 fixture 不存在：{reference}")
        return candidate

    @classmethod
    def from_file(cls, path: str | Path) -> "EvalScenario":
        source_path = Path(path).resolve()
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("场景 YAML 必须是对象")

        _reject_unknown_fields(
            payload,
            "scenario",
            {"id", "name", "description", "provenance", "execution", "scope", "tools", "expectations"},
        )
        provenance = _mapping(payload.get("provenance"), "provenance")
        _reject_unknown_fields(
            provenance,
            "provenance",
            {"kind", "source", "owner", "reviewed_at"},
        )
        execution = _mapping(payload.get("execution"), "execution")
        _reject_unknown_fields(
            execution,
            "execution",
            {"lane", "agent", "query", "max_tool_steps", "llm_mode", "replay"},
        )
        replay = _mapping(execution.get("replay"), "execution.replay")
        _reject_unknown_fields(
            replay,
            "execution.replay",
            {"tool_calls", "reference_answer"},
        )
        scope = _mapping(payload.get("scope"), "scope")
        _reject_unknown_fields(scope, "scope", {"mode", "redis_instance"})
        redis_instance = _mapping(scope.get("redis_instance"), "scope.redis_instance")
        _reject_unknown_fields(
            redis_instance,
            "scope.redis_instance",
            {
                "id",
                "name",
                "connection_url",
                "environment",
                "usage",
                "description",
                "status",
                "created_by",
            },
        )
        tools_root = _mapping(payload.get("tools"), "tools")
        _reject_unknown_fields(tools_root, "tools", _SUPPORTED_PROVIDER_FAMILIES)
        expectations = _mapping(payload.get("expectations"), "expectations")
        _reject_unknown_fields(
            expectations,
            "expectations",
            {"judge", "required_response_patterns", "required_findings", "forbidden_claims"},
        )

        parsed_calls: list[ReplayCall] = []
        for item in _list_of_mappings(replay.get("tool_calls"), "execution.replay.tool_calls"):
            _reject_unknown_fields(
                item,
                "execution.replay.tool_calls[]",
                {"provider", "operation", "args"},
            )
            parsed_calls.append(
                ReplayCall(
                    provider_family=_required_string(
                        item,
                        "execution.replay.tool_calls[].provider",
                    ),
                    operation=_required_string(item, "execution.replay.tool_calls[].operation"),
                    args=dict(_mapping(item.get("args", {}), "execution.replay.tool_calls[].args")),
                )
            )
        calls = tuple(parsed_calls)
        if not calls:
            raise ValueError("execution.replay.tool_calls 不能为空")

        tools: dict[str, dict[str, ToolBehavior]] = {}
        for provider_family, raw_provider_tools in tools_root.items():
            provider_tools = _mapping(raw_provider_tools, f"tools.{provider_family}")
            parsed_provider_tools: dict[str, ToolBehavior] = {}
            for operation, raw_behavior in provider_tools.items():
                behavior = _mapping(raw_behavior, f"tools.{provider_family}.{operation}")
                _reject_unknown_fields(
                    behavior,
                    f"tools.{provider_family}.{operation}",
                    {"result", "responders"},
                )
                parsed_responders: list[ToolResponder] = []
                for item in _list_of_mappings(behavior.get("responders", []), "responders"):
                    _reject_unknown_fields(item, "responder", {"when", "result"})
                    when = _mapping(item.get("when"), "responder.when")
                    _reject_unknown_fields(when, "responder.when", {"args_contains"})
                    parsed_responders.append(
                        ToolResponder(
                            args_contains=dict(
                                _mapping(
                                    when.get("args_contains", {}),
                                    "responder.when.args_contains",
                                )
                            ),
                            result=_required_string(item, "responder.result"),
                        )
                    )
                responders = tuple(parsed_responders)
                result = behavior.get("result")
                if result is not None and not isinstance(result, str):
                    raise ValueError(f"tools.{provider_family}.{operation}.result 必须是字符串")
                if result is None and not responders:
                    raise ValueError(
                        f"tools.{provider_family}.{operation} 必须配置 result 或 responders"
                    )
                parsed_provider_tools[str(operation)] = ToolBehavior(
                    result=result,
                    responders=responders,
                )
            tools[str(provider_family)] = parsed_provider_tools

        missing_operations = [
            f"{call.provider_family}.{call.operation}"
            for call in calls
            if call.provider_family not in tools
            or call.operation not in tools[call.provider_family]
        ]
        if missing_operations:
            raise ValueError(f"replay 调用了未声明工具：{', '.join(missing_operations)}")

        judge = str(expectations.get("judge") or "deterministic").strip().lower()
        if judge not in {"deterministic", "semantic"}:
            raise ValueError("expectations.judge 只支持 deterministic 或 semantic")

        execution_agent = _required_string(execution, "execution.agent")
        if execution_agent != "redis_sre":
            raise ValueError("execution.agent 当前只支持 redis_sre")
        scope_mode = _required_string(scope, "scope.mode")
        if scope_mode != "offline":
            raise ValueError("scope.mode 当前只支持 offline")

        return cls(
            id=_required_string(payload, "id"),
            name=_required_string(payload, "name"),
            description=str(payload.get("description") or "").strip(),
            provenance=ScenarioProvenance(
                kind=_required_string(provenance, "provenance.kind"),
                source=_required_string(provenance, "provenance.source"),
                owner=_required_string(provenance, "provenance.owner"),
                reviewed_at=_required_string(provenance, "provenance.reviewed_at"),
            ),
            execution_lane=str(execution.get("lane") or "agent_only").strip(),
            execution_agent=execution_agent,
            scope=ScenarioScope(
                mode=scope_mode,
                redis_instance=ScenarioRedisInstance(
                    id=_required_string(redis_instance, "scope.redis_instance.id"),
                    name=_required_string(redis_instance, "scope.redis_instance.name"),
                    connection_url=_required_string(
                        redis_instance,
                        "scope.redis_instance.connection_url",
                    ),
                    environment=_required_string(
                        redis_instance,
                        "scope.redis_instance.environment",
                    ),
                    usage=_required_string(redis_instance, "scope.redis_instance.usage"),
                    description=_required_string(
                        redis_instance,
                        "scope.redis_instance.description",
                    ),
                    status=_required_string(redis_instance, "scope.redis_instance.status"),
                    created_by=_required_string(
                        redis_instance,
                        "scope.redis_instance.created_by",
                    ),
                ),
            ),
            query=_required_string(execution, "query"),
            max_tool_steps=int(execution.get("max_tool_steps", 5)),
            llm_mode=_required_string(execution, "llm_mode"),
            judge=judge,
            replay_calls=calls,
            reference_answer=_required_string(replay, "reference_answer"),
            tools=tools,
            required_response_patterns=tuple(
                _strings(
                    expectations.get("required_response_patterns", []),
                    "required_response_patterns",
                )
            ),
            required_findings=tuple(_strings(expectations.get("required_findings", []), "required_findings")),
            forbidden_claims=tuple(_strings(expectations.get("forbidden_claims", []), "forbidden_claims")),
            _source_path=source_path,
        )


# 兼容已经引用旧名称的调用方；新 outcome 代码应使用 EvalScenario。
BigKeyScenario = EvalScenario


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是对象")
    return value


def _reject_unknown_fields(
    mapping: dict[str, Any],
    label: str,
    allowed_fields: set[str],
) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed_fields)
    if unknown:
        raise ValueError(f"{label} 包含未支持字段：{', '.join(unknown)}")


def _list_of_mappings(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label} 必须是对象列表")
    return value


def _required_string(mapping: dict[str, Any], label: str) -> str:
    value = mapping.get(label.rsplit(".", 1)[-1])
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    return value.strip()


def _strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{label} 必须是非空字符串列表")
    return [item.strip() for item in value]


def load_eval_scenario(path: str | Path) -> EvalScenario:
    return EvalScenario.from_file(path)


__all__ = [
    "BigKeyScenario",
    "EvalScenario",
    "ReplayCall",
    "ScenarioProvenance",
    "ScenarioRedisInstance",
    "ScenarioScope",
    "ToolBehavior",
    "ToolResponder",
    "load_eval_scenario",
]
