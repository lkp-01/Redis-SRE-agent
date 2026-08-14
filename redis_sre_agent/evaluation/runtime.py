"""运行 outcome 场景：真实 ChatAgent、确定性多 provider replay 和机械断言。"""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import Any, Sequence
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from redis_sre_agent.agent.chat_agent import ChatAgent
from redis_sre_agent.agent.helpers import guarded_ainvoke
from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.core.llm_helpers import create_llm
from redis_sre_agent.core.runtime_overrides import eval_tool_runtime_scope
from redis_sre_agent.evaluation.assertions import score_structured_assertions
from redis_sre_agent.evaluation.judge import (
    EvaluationResult,
    SREAgentJudge,
    evaluate_eval_scenario_response,
)
from redis_sre_agent.evaluation.report_schema import StructuredAssertionResults
from redis_sre_agent.evaluation.scenarios import (
    EvalScenario,
    ReplayCall,
    ScenarioScope,
    load_eval_scenario,
)
from redis_sre_agent.evaluation.tool_runtime import FixtureToolRuntime


class EvalRunResult(BaseModel):
    scenario_id: str
    scenario_name: str
    final_answer: str
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    structured_assertions: StructuredAssertionResults
    judge_result: EvaluationResult | None = None
    judge_passed: bool | None = None
    assertion_errors: list[str] = Field(default_factory=list)
    passed: bool | None


class ReplayToolCallingLLM:
    """按 YAML replay 工具调用，再让被测模型根据 ToolMessage 现场作答。"""

    def __init__(self, calls: Sequence[ReplayCall], answer_llm: Any) -> None:
        self._calls = tuple(calls)
        self._answer_llm = answer_llm
        self._tools: list[Any] = []

    def bind_tools(self, tools: Sequence[Any]) -> "ReplayToolCallingLLM":
        bound = copy(self)
        bound._tools = list(tools)
        return bound

    async def ainvoke(self, messages: Sequence[Any]) -> AIMessage:
        completed = sum(isinstance(message, ToolMessage) for message in messages)
        if completed >= len(self._calls):
            response = await guarded_ainvoke(
                self._answer_llm,
                [
                    *messages,
                    HumanMessage(
                        content=(
                            "All planned diagnostic evidence has been collected. "
                            "Produce the final answer from the ToolMessages above. "
                            "Do not call tools, do not invent evidence, distinguish "
                            "supported causes from uncertainty, and state which alternatives "
                            "the evidence supports or refutes."
                        )
                    ),
                ],
                request_kind="evaluation.answer_synthesis",
            )
            if not isinstance(response, AIMessage):
                response = AIMessage(content=getattr(response, "content", response))
            if response.tool_calls:
                raise RuntimeError("evaluation 回答模型不得继续调用工具")
            return response
        call = self._calls[completed]
        tool_name = self._tool_name(call)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"replay_{uuid4().hex[:10]}",
                    "name": tool_name,
                    "args": dict(call.args),
                }
            ],
        )

    def _tool_name(self, call: ReplayCall) -> str:
        suffix = f"_{call.operation}"
        for tool in self._tools:
            name = str(getattr(tool, "name", ""))
            if name.startswith(f"{call.provider_family}_") and name.endswith(suffix):
                return name
        raise RuntimeError(
            f"replay 找不到 {call.provider_family}.{call.operation} 工具"
        )


def _eval_instance(scope: ScenarioScope) -> RedisInstance:
    """把 scenario 声明的离线 Redis 目标转换成真实 Agent 使用的实例模型。"""

    if scope.mode != "offline":
        raise ValueError("scope.mode 当前只支持 offline")
    target = scope.redis_instance
    return RedisInstance(
        id=target.id,
        name=target.name,
        connection_url=target.connection_url,
        environment=target.environment,
        usage=target.usage,
        description=target.description,
        status=target.status,
        created_by=target.created_by,
    )


def _create_eval_agent(scenario: EvalScenario, answer_llm: Any) -> ChatAgent:
    """根据 execution.agent 创建场景声明的 Agent，拒绝静默回退。"""

    if scenario.execution_agent != "redis_sre":
        raise ValueError(f"execution.agent 不受支持：{scenario.execution_agent}")
    return ChatAgent(
        redis_instance=_eval_instance(scenario.scope),
        llm=ReplayToolCallingLLM(scenario.replay_calls, answer_llm),
    )


async def run_eval_scenario(
    scenario_or_path: EvalScenario | str,
    *,
    answer_llm: Any | None = None,
    judge: SREAgentJudge | None = None,
    judge_pass_threshold: float = 75.0,
) -> EvalRunResult:
    """运行 replay，并像 original 一样合并硬断言与 LLM judge 结果。"""

    scenario = (
        scenario_or_path
        if isinstance(scenario_or_path, EvalScenario)
        else load_eval_scenario(scenario_or_path)
    )
    if scenario.llm_mode != "replay":
        raise ValueError("当前最小 evaluation 只支持 llm_mode: replay")
    fixture_runtime = FixtureToolRuntime(scenario)
    agent = _create_eval_agent(scenario, answer_llm or create_llm())
    with eval_tool_runtime_scope(fixture_runtime):
        response = await agent.process_query(
            query=scenario.query,
            session_id=f"eval:{scenario.id}",
            user_id=None,
            max_iterations=scenario.max_tool_steps + 1,
        )
    tool_trace = [
        {
            "provider_family": trace.provider_family,
            "operation": trace.operation,
            "args": trace.args,
            "result": trace.result,
        }
        for trace in fixture_runtime.traces
    ]
    assertion_scenario = (
        replace(scenario, required_findings=(), forbidden_claims=())
        if scenario.judge == "semantic"
        else scenario
    )
    assertion_results = score_structured_assertions(
        assertion_scenario,
        tool_trace=tool_trace,
        final_answer=response.response,
    )
    grouped_results = [
        *assertion_results.required_tool_calls,
        *assertion_results.forbidden_tool_calls,
        *assertion_results.required_sources,
        *assertion_results.required_response_patterns,
        *assertion_results.forbidden_claims,
        *assertion_results.required_findings,
    ]
    errors = [item.message or "未命名断言失败" for item in grouped_results if not item.passed]
    judge_result = None
    judge_passed = None
    if scenario.judge == "semantic":
        judge_result = await evaluate_eval_scenario_response(
            scenario=scenario,
            agent_response=response.response,
            judge=judge or SREAgentJudge(),
            tool_trace=tool_trace,
            structured_assertions=assertion_results,
            diagnostic_data=[trace.result for trace in fixture_runtime.traces],
        )
        if judge_result.judge_valid:
            judge_passed = (
                judge_result.overall_score >= judge_pass_threshold
                and not judge_result.factual_errors
                and not judge_result.missing_elements
                and not judge_result.violated_forbidden_claims
            )
    if errors:
        passed: bool | None = False
    elif scenario.judge == "semantic" and judge_result is not None and not judge_result.judge_valid:
        passed = None
    else:
        passed = judge_passed is not False
    return EvalRunResult(
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        final_answer=response.response,
        tool_trace=tool_trace,
        structured_assertions=assertion_results,
        judge_result=judge_result,
        judge_passed=judge_passed,
        assertion_errors=errors,
        passed=passed,
    )


async def run_bigkey_scenario(
    scenario_or_path: EvalScenario | str,
    *,
    answer_llm: Any | None = None,
    judge: SREAgentJudge | None = None,
    judge_pass_threshold: float = 75.0,
) -> EvalRunResult:
    """兼容旧 BigKey 调用方，实际委托给通用 outcome runner。"""

    return await run_eval_scenario(
        scenario_or_path,
        answer_llm=answer_llm,
        judge=judge,
        judge_pass_threshold=judge_pass_threshold,
    )


__all__ = [
    "EvalRunResult",
    "ReplayToolCallingLLM",
    "run_bigkey_scenario",
    "run_eval_scenario",
]
