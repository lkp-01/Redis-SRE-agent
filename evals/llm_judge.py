from openevals.llm import create_llm_as_judge
from utils import SuccessAssertion,AgentTrajectory
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from langsmith import testing as t
import warnings
from dataclasses import dataclass, field

_TRAJECTORY_PROMPT = """
你是一个严格的评分助手。
你将收到一个智能体轨迹（一系列步骤）和一个单一的评估标准。
请判断该智能体的轨迹是否满足此标准。

每个步骤可能包含：
- 来自智能体的文本响应（显示为 "text: ..."）
- 智能体调用的工具（显示为 "- tool_name {{args}}"）

工具调用是智能体执行的真实操作。将其视为该操作已执行的证据

<criterion>
{criterion}
</criterion>

<agent_trajectory>
{outputs}
</agent_trajectory>"""

_RESPONSES_PROMPT = """
你是一个严格的评分助手。
你将收到一系列智能体的响应和一个评估标准。
请判断智能体的响应是否满足该评估标准。

<criterion>
{criterion}
</criterion>

<agent_responses>
{outputs}
</agent_responses>"""

_DEFAULT_JUDGE_MODEL = "deepseek-v4-pro"
_MAX_JUDGE_WORKERS = 8

@dataclass
class LLMJudge(SuccessAssertion):

#-----------------------------------------
# _grade的函数，按照顺序来
#-----------------------------------------
    def _serialize(self, trajectory: AgentTrajectory) -> str:
        """将 AgentTrajectory 对象转化为字符串格式（序列化）"""
        if self.include_tool_calls:
            return trajectory.pretty()

        return "\n\n".join(
            f"[Agent]: {step.action.text}" for step in trajectory.steps if step.action.text
        )

    include_tool_calls: bool = False
    """是否在发送给裁判的上下文中包含工具调用。"""

    judge_model: str = _DEFAULT_JUDGE_MODEL
    """裁判大语言模型的模型标识符。"""

    criteria: tuple[str, ...]= field(default_factory=tuple)
    """智能体输出必须满足的人类可读的评估标准。"""

#·········································································································

    def _grade(self, trajectory: AgentTrajectory) -> list[dict[str, Any]]:
        """调用 openevals 裁判对每个评估标准进行评分并返回结果。

        参数:
            trajectory: 要评分的智能体轨迹。

        返回:
            `EvaluatorResult` 字典列表，每个标准对应一个。
        """
        conversation = self._serialize(trajectory)
        if not conversation.strip():
            msg = (
                "无法对轨迹进行评分：没有任何步骤包含内容。"
                "LLM 裁判至少需要一个步骤来进行评估。"
            )
            raise ValueError(msg)

        prompt = _TRAJECTORY_PROMPT if self.include_tool_calls else _RESPONSES_PROMPT
        evaluator = create_llm_as_judge(
            prompt=prompt,
            feedback_key="llm_judge_criterion",  #起了一个名字
            model=self.judge_model,
        )

        def _evaluate(criterion: str) -> dict[str, Any]:
            return evaluator(outputs=conversation, criterion=criterion)

        # 每个评估标准的裁判调用都是独立的网络调用，因此采用并发运行，
        # 而不是对每个标准串行进行一次往返调用。所有标准都会被评估（在第一个
        # 失败项上不会提前短路）；结果会按标准索引存储并按顺序消费，
        # 这样无论完成顺序如何，抛出的错误始终是确定的。
        total = len(self.criteria)
        raw_results: list[Any] = [None] * total
        if total > 1:
            with ThreadPoolExecutor(max_workers=min(total, _MAX_JUDGE_WORKERS)) as executor:
                futures = {}
                for idx, criterion in enumerate(self.criteria):
                    ctx = copy_context()
                    futures[executor.submit(ctx.run, _evaluate, criterion)] = idx
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        raw_results[idx] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        raw_results[idx] = exc
        else:
            for idx, criterion in enumerate(self.criteria):
                try:
                    raw_results[idx] = _evaluate(criterion)
                except Exception as exc:  # noqa: BLE001
                    raw_results[idx] = exc

        results: list[dict[str, Any]] = []
        for i, (criterion, result) in enumerate(zip(self.criteria, raw_results, strict=True), 1):
            if isinstance(result, BaseException):
                msg = (
                    f"LLM 裁判在评估标准 {i}/{total} 时失败 "
                    f"(model={self.judge_model!r}): {criterion!r}"
                )
                raise RuntimeError(msg) from result  # noqa: TRY004
            if not isinstance(result, dict) or "score" not in result:
                msg = (
                    f"openevals 对评估标准返回了意外的结果 "
                    f"{i}/{total} {criterion!r}: {result!r}"
                )
                raise ValueError(msg)
            results.append(result)

        # 将聚合的裁判结果记录到 LangSmith。
        passed = sum(1 for r in results if r["score"])
        try:
            t.log_feedback(
                key="llm_judge_all_passed",
                score=1.0 if passed == len(results) else 0.0,
                comment=f"{passed}/{len(results)} 个评估标准已通过",
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"未能将 LLM 裁判反馈记录到 LangSmith: {type(exc).__name__}: {exc}",
                stacklevel=2,
            )

        return results

#-----------------------------------------
# 实现父类函数
#-----------------------------------------
    def check(self, trajectory: AgentTrajectory) -> bool:
        """调用 LLM Judge，所有 criteria 都通过时返回 True。"""
        results = self._grade(trajectory)

        # 缓存本次 Judge 结果，避免 describe_failure() 再调用一次 LLM
        self._last_results = results

        return all(result["score"] for result in results)


    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """返回所有失败 criterion 的具体原因。"""

        # 正常情况下 check() 已经跑过，直接复用结果
        results = (
            self._last_results
            if self._last_results is not None
            else self._grade(trajectory)
        )

        failed = [
            (index, result)
            for index, result in enumerate(results, start=1)
            if not result["score"]
        ]

        parts = [
            f"Criterion {index} failed: "
            f"{result.get('comment') or 'LLM Judge 未提供失败原因'}"
            for index, result in failed
        ]

        return (
            f"{len(failed)}/{len(results)} criteria failed — "
            + "; ".join(parts)
        )






def llm_judge(
    *criteria: str,
    judge_model: str = _DEFAULT_JUDGE_MODEL,
    include_tool_calls: bool = False,
) -> LLMJudge:
    """创建一个 `LLMJudge` 成功断言。

    封装 `openevals.llm.create_llm_as_judge`，针对智能体的输出独立评估每个评估标准。
    所有标准都必须通过，断言才能成功。

    参数:
        *criteria: 一个或多个人类可读的评估标准字符串。
        judge_model: 裁判大语言模型的模型标识符。
        include_tool_calls: 如果为 True，裁判将查看完整的轨迹（工具调用 + 文本）。
            
            如果为 False（默认），则仅查看文本响应。

    返回:
        一个 `LLMJudge` 断言实例。

    引发:
        ValueError: 如果未提供任何评估标准。
    """
    return LLMJudge(
        criteria=tuple(criteria),
        judge_model=judge_model,
        include_tool_calls=include_tool_calls,
    )