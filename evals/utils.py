from __future__ import annotations
from dataclasses import dataclass
from langchain_core.messages import AIMessage, ToolMessage, AnyMessage, HumanMessage
from collections.abc import Mapping, Sequence, Callable
from langsmith import testing as t  #langsmith的平台
from typing import Any, TYPE_CHECKING
from redis_sre_agent.agent.models import AgentResponse

import logging
from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)
# from deepagents.backends.utils import create_file_data, file_data_to_string

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

import uuid
import pytest

@dataclass(frozen=True)
class AgentStep:
    index: int
    action: AIMessage
    observations: list[ToolMessage]

    def __post_init__(self) -> None:
            if self.index <= 0:
                msg = "index must be positive"
                raise ValueError(msg)

@dataclass(frozen=True)
class AgentTrajectory:
    """
    一次 Redis SRE Agent eval 的完整执行轨迹。

    这是 eval harness 的统一数据模型。
    后面的 correctness、tool-use、retrieval、safety、
    efficiency 等评测都只依赖这个对象。
    """

    # 最终给用户的回答。
    answer: str

    # 当前这一轮 Agent 的完整消息轨迹。
    # 包括 HumanMessage / AIMessage / ToolMessage 等。
    messages: list[AnyMessage]

    # 从 messages 中整理出的 Agent 思考-工具循环。
    steps: list[AgentStep]

    # Redis / Prometheus / Loki / RAG 等真实工具调用产生的证据。
    # 每个元素对应项目里的 ResultEnvelope。
    tool_envelopes: list[dict[str, Any]]

    # RAG citation / search 结果。
    search_results: list[dict[str, Any]]

    # 当前 Agent loop 实际运行了多少轮。
    iteration_count: int

    # Agent 执行层是否发生异常。
    # 正常情况为 None。
    error: str | None = None
    
    def pretty(self) -> str:
        """提取每一步的摘要，工具调用和文本输出"""
        lines: list[str] = []
        for step in self.steps:
            lines.append(f"step {step.index}:")
            tool_calls = step.action.tool_calls
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name")
                    args = tc.get("args")
                    lines.append(f"  - {name} {args}")
            text = step.action.text
            if text and text.strip():
                text_preview = text.strip().replace("\n", "\\n")
                lines.append(f"  text: {text_preview}")
        return "\n".join(lines)

# ---------------------------------------------------------------------------
# 此处开始实现 _assert_expectations 中的函数和class
# 1、TrajectoryScorer + SuccessAssertion、EfficiencyAssertion、ToolCall
# 2、两个efficiency + toolcall、_fina_tool_call_matched
# ---------------------------------------------------------------------------
# 1

#下面俩都是基类，所以没有具体的代码
class SuccessAssertion:
    """当违反断言时导致测试失败的正确性断言的基类。"""

    def check(self, trajectory: AgentTrajectory) -> bool:
        """当断言成立时返回 `True`。

        参数:
            trajectory: 要检查的智能体轨迹 (agent trajectory)。

        返回:
            断言是否通过。

        引发:
            NotImplementedError: 子类必须重写此方法。
        """
        raise NotImplementedError

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """返回关于检查为何失败的易于理解（人类可读）的解释。

        参数:
            trajectory: 未通过检查的智能体轨迹。

        返回:
            关于失败原因的描述。

        引发:
            NotImplementedError: 子类必须重写此方法。
        """
        raise NotImplementedError
        
@dataclass(frozen=True)
class EfficiencyAssertion:
    """轨迹形状断言的基类，这些断言会被记录，但永远不会导致失败。"""

    def check(self, trajectory: AgentTrajectory) -> bool:
        """当断言成立时返回 `True`。

        参数:
            trajectory: 要检查的智能体轨迹 (agent trajectory)。

        返回:
            断言是否通过。

        引发:
            NotImplementedError: 子类必须重写此方法。
        """
        raise NotImplementedError

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """返回关于检查为何失败的易于理解（人类可读）的解释。

        参数:
            trajectory: 未通过检查的智能体轨迹。

        返回:
            关于失败原因的描述。

        引发:
            NotImplementedError: 子类必须重写此方法。
        """
        raise NotImplementedError

@dataclass(frozen=True)
class AgentSteps(EfficiencyAssertion):
    n: int

    def check(self, trajectory: AgentTrajectory) -> bool:
        return len(trajectory.steps) == self.n

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        return f"Expected {self.n} agent steps, got {len(trajectory.steps)}"

@dataclass(frozen=True)
class ToolCall(EfficiencyAssertion):
    """断言轨迹中发生过特定的工具调用。

    当 `step` 为 `None` 时，将搜索所有步骤。当给定 `step` 时，
    仅检查该步骤（基于 1 的索引）。

    属性:
        name: 预期的工具名称。
        step: 可选的基于 1 的步骤索引，用于限制搜索范围。
        args_contains: 如果设置，工具调用参数必须包含这些键值对。
        args_equals: 如果设置，工具调用参数必须与此字典完全相等。
    """

    name: str
    step: int | None = None
    args_contains: dict[str, object] | None = None
    args_equals: dict[str, object] | None = None

    def __post_init__(self) -> None:
        """在构造时拒绝非正数的步骤索引。"""
        if self.step is not None and self.step <= 0:
            msg = f"step must be positive (1-indexed), got {self.step}"
            raise ValueError(msg)

    def check(self, trajectory: AgentTrajectory) -> bool:
        """检查轨迹中是否存在匹配的工具调用。

        参数:
            trajectory: 要检查的智能体轨迹。

        返回:
            是否找到匹配的工具调用。
        """
        return bool(
            _find_tool_call_matches(
                trajectory,
                name=self.name,
                step=self.step,
                args_contains=self.args_contains,
                args_equals=self.args_equals,
            )
        )

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """描述工具调用检查为何失败。

        参数:
            trajectory: 未通过检查的智能体轨迹。

        返回:
            易于理解（人类可读）的失败描述。
        """
        step_desc = f" in step {self.step}" if self.step is not None else ""
        return f"Missing expected tool call{step_desc}: name={self.name!r}, args_contains={self.args_contains!r}, args_equals={self.args_equals!r}"


@dataclass(frozen=True)
class TrajectoryScorer:
    """智能体执行轨迹的双层断言容器。

    使用 `.success()` 来添加正确性断言（违反时会触发硬性失败），
    使用 `.expect()` 来添加效率断言（仅记录日志，绝不会导致测试失败）。

    属性 (Attributes):
        _success: 成功（正确性）断言的元组。
        _expectations: 效率断言的元组。
    """
    _success: tuple[SuccessAssertion, ...] = ()
    _expectations: tuple[EfficiencyAssertion, ...] = ()

    def success(self, *assertions: SuccessAssertion) -> TrajectoryScorer:
        """追加正确性断言，当违反这些断言时将导致测试硬性失败。

        参数 (Args):
            *assertions: 一个或多个 `SuccessAssertion` 实例。

        返回 (Returns):
            一个追加了新断言的全新 `TrajectoryScorer` 对象。

        本质就是对正确性断言规则的一次更新（运行之前）
        """
        return TrajectoryScorer(
            _success=(*self._success, *assertions),
            _expectations=self._expectations,
        )

    def expect(
        self,
        *,
        agent_steps: int | None = None,
        tool_call_requests: int | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> TrajectoryScorer:
        """追加效率断言，这些断言会被记录在日志中，但绝不会导致测试失败。

        参数 (Args):
            agent_steps: 预期的智能体执行步数。
            tool_call_requests: 预期的工具调用总次数。
            tool_calls: 预期的具体工具调用记录（可选择绑定到具体的步骤上）。

        返回 (Returns):
            一个追加了新断言的全新 `TrajectoryScorer` 对象。

        本质就是对效率断言规则的一次更新（运行之前）
        """
        new: list[EfficiencyAssertion] = []
        if agent_steps is not None:
            new.append(AgentSteps(n=agent_steps))
        if tool_call_requests is not None:
            new.append(ToolCallRequests(n=tool_call_requests))
        if tool_calls is not None:
            new.extend(tool_calls)
        return TrajectoryScorer(
            _success=self._success,
            _expectations=(*self._expectations, *new),
        )


# 2

def _tool_call_matches(
    tc: Mapping[str, object],
    *,
    name: str,
    args_contains: dict[str, object] | None,
    args_equals: dict[str, object] | None,
) -> bool:
    """检查单个工具调用字典是否与选择器匹配。

    参数:
        tc: 包含 `name` 和 `args` 键的工具调用字典。
        name: 预期的工具名称。
        args_contains: 如果设置，参数必须包含这些键值对。
        args_equals: 如果设置，参数必须与此字典完全相等。

    返回:
        该工具调用是否匹配。
    """
    if tc.get("name") != name:
        return False
    if args_contains is not None:
        args = tc.get("args")
        if not isinstance(args, dict):
            return False
        if not all(k in args and args.get(k) == v for k, v in args_contains.items()):
            return False
    return args_equals is None or tc.get("args") == args_equals

def _find_tool_call_matches(
    trajectory: AgentTrajectory,
    *,
    name: str,
    step: int | None,
    args_contains: dict[str, object] | None,
    args_equals: dict[str, object] | None,
) -> list[Mapping[str, object]]:
    """在 `trajectory`（轨迹）中查找与选择器匹配的工具调用。

    当 `step` 为 `None` 时，将搜索所有步骤。当给定 `step` 时，仅
    检查该步骤（基于 1 的索引）。由 `ToolCall`（存在性检查）和
    `ToolNotCalled`（缺席/未调用检查）共享，以确保两者保持逻辑同步。

    参数:
        trajectory: 要搜索的智能体轨迹。
        name: 预期的工具名称。
        step: 可选的基于 1 的步骤索引，用于限制搜索范围。
        args_contains: 如果设置，参数必须包含这些键值对。
        args_equals: 如果设置，参数必须与此字典完全相等。

    返回:
        匹配的工具调用字典列表。
    """
    if step is not None:
        if step > len(trajectory.steps):
            return []
        steps_to_search = [trajectory.steps[step - 1]]
    else:
        steps_to_search = trajectory.steps
    return [
        tc
        for s in steps_to_search
        for tc in s.action.tool_calls
        if _tool_call_matches(tc, name=name, args_contains=args_contains, args_equals=args_equals)
    ]


@dataclass
class EfficiencyResult:
    """每一个测试收集到的效率数据"""

    expected_steps: int | None
    actual_steps: int
    expected_tool_calls: int | None
    actual_tool_calls: int
    duration_s: float | None = None
    passed: bool | None = None

@dataclass(frozen=True)
class ToolCallRequests(EfficiencyAssertion):
    """断言轨迹中包含确切数量为 `n` 的工具调用请求总数。

    属性:
        n: 预期的工具调用请求总数。
    """

    n: int

    def check(self, trajectory: AgentTrajectory) -> bool:
        """检查工具调用请求总数是否等于 `self.n`。

        参数:
            trajectory: 要检查的智能体轨迹。

        返回:
            工具调用数量是否匹配。
        """
        actual = sum(len(s.action.tool_calls) for s in trajectory.steps)
        return actual == self.n

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """描述工具调用请求检查为何失败。

        参数:
            trajectory: 未通过检查的智能体轨迹。

        返回:
            易于理解（人类可读）的失败描述。
        """
        actual = sum(len(s.action.tool_calls) for s in trajectory.steps)
        return f"Expected {self.n} tool call requests, got {actual}"

def _log_efficiency(
    trajectory: AgentTrajectory,
    scorer: TrajectoryScorer,
) -> EfficiencyResult | None:
    """将效率反馈记录到 LangSmith 并返回收集到的数据。

    参数:
        trajectory: 智能体轨迹 (agent trajectory)。
        scorer: 包含效率预期的评分器。

    返回:
        当评分器具有步骤或工具调用预期时，返回 `EfficiencyResult`，否则返回 `None`。
    """
    actual_steps = len(trajectory.steps)
    actual_tool_calls = sum(len(s.action.tool_calls) for s in trajectory.steps)
    t.log_feedback(key="agent_steps", value=actual_steps)
    t.log_feedback(key="tool_call_requests", value=actual_tool_calls)

    expected_steps: int | None = None
    expected_tool_calls: int | None = None
    for assertion in scorer._expectations:
        if isinstance(assertion, AgentSteps):
            expected_steps = assertion.n
        elif isinstance(assertion, ToolCallRequests):
            expected_tool_calls = assertion.n

    if expected_steps is not None:
        t.log_feedback(key="expected_agent_steps", value=expected_steps)
    if expected_tool_calls is not None:
        t.log_feedback(key="expected_tool_call_requests", value=expected_tool_calls)

    if expected_steps is None and expected_tool_calls is None:
        return None

    return EfficiencyResult(
        expected_steps=expected_steps,
        actual_steps=actual_steps,
        expected_tool_calls=expected_tool_calls,
        actual_tool_calls=actual_tool_calls,
    )

_on_efficiency_result: Callable[[EfficiencyResult], None] | None = None
"""由报告器 (reporter) 插件设置的回调，用于收集每次测试的效率数据。"""

# ---------------------------------------------------------------------------
# 此处开始run_agent中的函数（除了assertaion_expectation）调用的函数
# ---------------------------------------------------------------------------

# def _coerce_result_files_to_strings(raw_files: object) -> dict[str, str]:


# ---------------------------------------------------------------------------
# 此处开始实现run_agent中的具体的函数，从上到下挨个来
# ---------------------------------------------------------------------------



def _build_logged_inputs(
    model: BaseChatModel,
    eval_metadata: dict[str, object] | None,
) -> dict[str, Any]:
    """创建本次运行时langsmith抓取的数据，包括信息标识（run_tree.name）、大模型信息和元数据，准备下一步替换"""
    run_tree = get_current_run_tree()
    model_str = str(getattr(model, "model", None) or getattr(model, "model_name", ""))
    logged_inputs: dict[str, Any] = {
        "test_name": run_tree.name if run_tree else "unknown",
        "model": model_str,
    }
    if eval_metadata is not None:
        logged_inputs["eval_metadata"] = eval_metadata
    return logged_inputs

def _log_run_inputs(logged_inputs: dict[str, Any]) -> None:
    """用刚刚打包的数据，替换langsmith自己要抓取的数据"""
    t.log_inputs(logged_inputs)
    run_tree = get_current_run_tree()
    if run_tree is not None:
        run_tree.inputs = logged_inputs
    else:
        logger.debug(
            "run_tree为空; run_tree.inputs 不会被覆盖 "
            "(sync_example may record auto-captured inputs)"
        )

def _trajectory_from_result(
    result: AgentResponse
) -> AgentTrajectory:
    if result.trace is None:
        raise ValueError("AgentResponse does not contain execution trace")

    messages = list(result.trace.messages)
    iteration_count = result.trace.iteration_count

    # 只取当前这一轮。
    last_human_index = -1
    for i, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human_index = i

    current_messages = (
        messages[last_human_index:]
        if last_human_index >= 0
        else messages
    )

    # 先把 AIMessage + 后续 ToolMessage 分组。
    raw_steps: list[tuple[AIMessage, list[ToolMessage]]] = []

    current_action: AIMessage | None = None
    current_observations: list[ToolMessage] = []

    for message in current_messages:
        if isinstance(message, AIMessage):
            if current_action is not None:
                raw_steps.append(
                    (current_action, current_observations)
                )

            current_action = message
            current_observations = []

        elif isinstance(message, ToolMessage):
            if current_action is not None:
                current_observations.append(message)

    if current_action is not None:
        raw_steps.append(
            (current_action, current_observations)
        )

    # 关键：
    # triage reasoning_node 可能额外产生 final AIMessage，
    # 它不属于 agent loop，因此只取 iteration_count 个。
    if len(raw_steps) < iteration_count:
        raise ValueError(
            f"Trace contains {len(raw_steps)} AI steps, "
            f"but iteration_count={iteration_count}"
        )

    steps = [
        AgentStep(
            index=i + 1,
            action=action,
            observations=observations,
        )
        for i, (action, observations)
        in enumerate(raw_steps[:iteration_count])
    ]

    return AgentTrajectory(
        answer=result.response,
        messages=current_messages,
        steps=steps,
        tool_envelopes=list(result.tool_envelopes),
        search_results=list(result.search_results),
        iteration_count=iteration_count,
        error=result.trace.error,
    )

def _assert_expectations(
    trajectory: AgentTrajectory,
    scorer: TrajectoryScorer,
) -> None:
    """针对 *trajectory* 运行 *scorer* 中的所有断言。

    成功断言（Success assertions）会通过 `pytest.fail` 导致测试直接失败。效率
    断言（Efficiency assertions）会作为反馈记录，但绝不会导致测试失败。

    参数:
        trajectory: 要验证的智能体轨迹。
        scorer: 两层预期容器。
    """
    eff_result = _log_efficiency(trajectory, scorer)
    if eff_result is not None and _on_efficiency_result is not None:
        _on_efficiency_result(eff_result)  #上面是系统检查有没有设置手机效率数据的回调函数。这里是调用该回调函数。

    # 硬正确性检查
    success = True
    for assertion in scorer._success:
        if not assertion.check(trajectory):
            success = False
            t.log_feedback(key="correctness", value=0)
            pytest.fail(
                f"success check failed: {assertion.describe_failure(trajectory)}\n\ntrajectory:\n{trajectory.pretty()}",
                pytrace=False,
            )
    if success:
        t.log_feedback(key="correctness", value=1)

#===========================================================================
# 根据我自己的修改
#===========================================================================
@dataclass
class EvalEnvironment:
    redis_data: dict[str, Any]
    metrics: dict[str, Any] | None = None
    logs: list[dict[str, Any]] | None = None

#===========================================================================

async def run_agent_async(
    agent: Any,
    *,
    query: str | Sequence[AnyMessage],
    session_id: str = "eval",
    user_id: str | None = None,
    max_iterations: int = 10,
    context: dict[str, Any] | None = None,
    conversation_history: list[AnyMessage] | None = None,
    model: BaseChatModel,
    environment: EvalEnvironment,
    scorer: TrajectoryScorer | None = None,
    eval_metadata: dict[str, object] | None = None,
) -> AgentTrajectory:

    logged_inputs = _build_logged_inputs(model, eval_metadata)
    _log_run_inputs(logged_inputs)

    result = await agent.process_query(
        query,
        session_id=session_id,
        user_id=user_id,
        max_iterations=max_iterations,
        context=context,
        conversation_history=conversation_history,
        capture_trace=True,
    )

    t.log_outputs(result.model_dump())

    trajectory = _trajectory_from_result(result)

    if scorer is not None:
        _assert_expectations(trajectory, scorer)

    return trajectory