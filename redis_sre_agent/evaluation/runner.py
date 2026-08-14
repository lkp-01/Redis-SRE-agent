"""Outcome 单场景的公共运行入口。"""

from __future__ import annotations

from redis_sre_agent.evaluation.runtime import EvalRunResult, run_eval_scenario as _run_eval_scenario


async def run_eval_scenario(path: str) -> EvalRunResult:
    return await _run_eval_scenario(path)


__all__ = ["run_eval_scenario"]
