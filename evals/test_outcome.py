"""Redis SRE 诊断结果的评估测试。
测试智能体是否针对 Redis 故障得出正确且有据可查的诊断结论。
"""
from __future__ import annotations

import pytest

from evals.utils import (
    TrajectoryScorer,
    run_agent,
)

pytestmark = [
    pytest.mark.eval_category("outcome"),
]


# ============================================================
# A：Baseline outcome evals
# ============================================================

# @pytest.mark.eval_tier("baseline")
# def test_xxx(...):
#     """
#     这里描述这个 eval 想验证什么能力。
#     """

#     # setup
#     ...

#     # execute + evaluate
#     run_agent(
#         ...,
#         scorer=(
#             TrajectoryScorer()
#             ...
#         ),
#     )


# ============================================================
# B：Hillclimb outcome evals
# ============================================================