"""提供 outcome 离线 replay、确定性硬断言和可注入的 LLM judge。

knowledge、MCP、memory、live suite、retrieval 和完整报告产物仍是后续阶段插槽。
"""

from redis_sre_agent.evaluation.runner import run_eval_scenario
from redis_sre_agent.evaluation.runtime import EvalRunResult

__all__ = ["EvalRunResult", "run_eval_scenario"]
