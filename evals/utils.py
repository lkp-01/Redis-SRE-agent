from __future__ import annotations
from collections.abc import AsyncIterator, Mapping, Sequence, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import ipaddress
import logging
import os
from typing import Any, TYPE_CHECKING
from urllib.parse import urlparse
import uuid

from langchain_core.messages import AIMessage, ToolMessage, AnyMessage, HumanMessage
from langsmith import testing as t  #langsmith的平台
from pydantic import SecretStr
from redis.asyncio import Redis
from redis_sre_agent.agent.models import AgentResponse
from redis_sre_agent.core.instances import RedisInstance

from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)
# from deepagents.backends.utils import create_file_data, file_data_to_string

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
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
    try:
        t.log_inputs(logged_inputs)
    except ValueError:
        logger.debug("LangSmith test context is unavailable; eval inputs will not be recorded")
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
    """一个 eval case 声明的、可被物化的 Redis 世界。"""

    redis_data: dict[str, Any]
    metrics: dict[str, Any] | None = None
    logs: list[dict[str, Any]] | None = None
    redis_config: dict[str, str | int | float] = field(default_factory=dict)
    redis_setup_commands: list[tuple[str, list[Any]]] = field(default_factory=list)


@dataclass(frozen=True)
class EvalRuntime:
    """`EvalEnvironment` 物化后实际提供给 Agent 的运行时资源。"""

    redis_instance: RedisInstance
    redis_url: str
    # Phase 2: materialize metrics/logs into isolated Prometheus/Loki test runtime.
    prometheus_url: str | None = None
    loki_url: str | None = None


_EVAL_REDIS_URL_ENV = "EVAL_REDIS_URL"
_SUPPORTED_REDIS_CONFIG_KEYS = frozenset(
    {
        "maxmemory",
        "maxmemory-policy",
        "timeout",
        "slowlog-log-slower-than",
        "slowlog-max-len",
    }
)
_ALLOWED_SETUP_COMMANDS = frozenset(
    {
        "SET",
        "MSET",
        "HSET",
        "LPUSH",
        "RPUSH",
        "SADD",
        "ZADD",
        "XADD",
        "GEOADD",
        "PFADD",
        "SETBIT",
    }
)


def _is_loopback_redis_host(hostname: str | None) -> bool:
    """只将 loopback 主机视为默认可安全清空的 eval Redis。"""

    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _resolve_eval_redis_url(redis_url: str | None = None) -> str:
    """读取并验证专用 eval Redis URL，拒绝任何非 loopback 目标。"""

    candidate = (redis_url or os.getenv(_EVAL_REDIS_URL_ENV) or "").strip()
    if not candidate:
        raise RuntimeError(
            "Eval Redis is not configured. Set EVAL_REDIS_URL to a dedicated "
            "loopback Redis database before running Redis evals."
        )

    parsed = urlparse(candidate)
    if parsed.scheme not in {"redis", "rediss"} or not _is_loopback_redis_host(parsed.hostname):
        raise ValueError(
            "EVAL_REDIS_URL must target a dedicated loopback Redis instance; "
            "non-loopback and unix-socket URLs are refused because eval teardown uses FLUSHDB."
        )
    return candidate


def _coerce_redis_scalar(value: Any, *, label: str) -> str | int | float | bytes:
    """限制声明式 seed 数据为 redis-py 可安全表达的标量。"""

    if isinstance(value, bool) or not isinstance(value, (str, int, float, bytes)):
        raise TypeError(
            f"{label} must be a str, int, float, or bytes; got {type(value).__name__}"
        )
    return value


async def _seed_redis_data(client: Redis, redis_data: Mapping[str, Any]) -> None:
    """将简单的 Python 数据结构写入本 case 的专用 Redis 数据库。"""

    for raw_key, value in redis_data.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError("redis_data keys must be non-empty strings")

        if isinstance(value, Mapping):
            if not value:
                raise ValueError(f"redis_data[{raw_key!r}] cannot be an empty hash")
            mapping = {
                field_name: _coerce_redis_scalar(
                    field_value,
                    label=f"redis_data[{raw_key!r}][{field_name!r}]",
                )
                for field_name, field_value in value.items()
            }
            if not all(isinstance(field_name, str) and field_name for field_name in mapping):
                raise ValueError(f"redis_data[{raw_key!r}] hash fields must be non-empty strings")
            await client.hset(raw_key, mapping=mapping)
        elif isinstance(value, list):
            if not value:
                raise ValueError(f"redis_data[{raw_key!r}] cannot be an empty list")
            await client.rpush(
                raw_key,
                *(
                    _coerce_redis_scalar(item, label=f"redis_data[{raw_key!r}] list item")
                    for item in value
                ),
            )
        else:
            await client.set(raw_key, _coerce_redis_scalar(value, label=f"redis_data[{raw_key!r}]"))


async def _apply_redis_setup_commands(
    client: Redis,
    commands: Sequence[tuple[str, list[Any]]],
) -> None:
    """运行少量无法由 ``redis_data`` 表达的、白名单内的写入初始化命令。"""

    for raw_command, raw_args in commands:
        command = str(raw_command).upper()
        if command not in _ALLOWED_SETUP_COMMANDS:
            raise ValueError(
                f"redis_setup_commands does not allow {command!r}; "
                f"allowed commands: {sorted(_ALLOWED_SETUP_COMMANDS)}"
            )
        if not isinstance(raw_args, list):
            raise TypeError(f"redis_setup_commands arguments for {command!r} must be a list")
        await client.execute_command(command, *raw_args)


async def _capture_redis_config(
    client: Redis,
    redis_config: Mapping[str, str | int | float],
) -> dict[str, str]:
    """读取每个将被覆盖的 Redis 配置，以便 finally 中无条件恢复。"""

    original: dict[str, str] = {}
    for raw_key, value in redis_config.items():
        if raw_key not in _SUPPORTED_REDIS_CONFIG_KEYS:
            raise ValueError(
                f"redis_config key {raw_key!r} is unsupported; "
                f"supported keys: {sorted(_SUPPORTED_REDIS_CONFIG_KEYS)}"
            )
        _coerce_redis_scalar(value, label=f"redis_config[{raw_key!r}]")
        current = await client.config_get(raw_key)
        saved_value = current.get(raw_key)
        if saved_value is None:
            raise RuntimeError(f"Eval Redis did not return current value for CONFIG GET {raw_key}")
        original[raw_key] = str(saved_value)
    return original


async def _restore_redis_config(client: Redis, original_config: Mapping[str, str]) -> None:
    """尽最大努力恢复本 case 覆盖过的配置。"""

    for key, value in original_config.items():
        await client.config_set(key, value)


@asynccontextmanager
async def materialize_environment(
    environment: EvalEnvironment,
    *,
    redis_url: str | None = None,
) -> AsyncIterator[EvalRuntime]:
    """将一个声明式 eval 环境物化到专用的 loopback Redis，并在退出时完整清理。

    该实现用 ``FLUSHDB`` 隔离 case；共享同一 ``EVAL_REDIS_URL`` 的 eval 必须串行执行。
    """

    resolved_url = _resolve_eval_redis_url(redis_url)
    client = Redis.from_url(resolved_url, decode_responses=True)
    original_config: dict[str, str] = {}

    try:
        try:
            await client.ping()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Eval Redis is unavailable. Verify that EVAL_REDIS_URL points to a reachable "
                "dedicated loopback Redis instance."
            ) from exc

        await client.flushdb()
        original_config = await _capture_redis_config(client, environment.redis_config)
        for key, value in environment.redis_config.items():
            await client.config_set(key, str(value))
        await _seed_redis_data(client, environment.redis_data)
        await _apply_redis_setup_commands(client, environment.redis_setup_commands)

        runtime = EvalRuntime(
            redis_instance=RedisInstance(
                id=f"eval-redis-{uuid.uuid4().hex}",
                name="Redis SRE eval environment",
                connection_url=SecretStr(resolved_url),
                environment="test",
                usage="eval",
                description="Isolated Redis SRE eval environment",
                created_by="agent",
            ),
            redis_url=resolved_url,
        )
        yield runtime
    finally:
        try:
            await client.flushdb()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to clean up the eval Redis database")
        try:
            await _restore_redis_config(client, original_config)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to restore eval Redis configuration")
        await client.aclose()


@asynccontextmanager
async def _inject_eval_runtime(agent: Any, runtime: EvalRuntime) -> AsyncIterator[None]:
    """临时把真实 Agent 路由到 eval Redis，并精确恢复原始属性。"""

    missing = object()
    original_instance = getattr(agent, "redis_instance", missing)
    original_cluster = getattr(agent, "redis_cluster", missing)
    setattr(agent, "redis_instance", runtime.redis_instance)
    setattr(agent, "redis_cluster", None)
    try:
        yield
    finally:
        if original_instance is missing:
            delattr(agent, "redis_instance")
        else:
            setattr(agent, "redis_instance", original_instance)
        if original_cluster is missing:
            delattr(agent, "redis_cluster")
        else:
            setattr(agent, "redis_cluster", original_cluster)

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

    async with materialize_environment(environment) as runtime:
        async with _inject_eval_runtime(agent, runtime):
            result = await agent.process_query(
                query,
                session_id=session_id,
                user_id=user_id,
                max_iterations=max_iterations,
                context=context,
                conversation_history=conversation_history,
                capture_trace=True,
            )

    try:
        t.log_outputs(result.model_dump())
    except ValueError:
        logger.debug("LangSmith test context is unavailable; eval outputs will not be recorded")

    trajectory = _trajectory_from_result(result)

    if scorer is not None:
        _assert_expectations(trajectory, scorer)

    return trajectory
