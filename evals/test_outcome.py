"""Redis SRE 诊断结果的评估测试。
测试智能体是否针对 Redis 故障得出正确且有据可查的诊断结论。
"""
from __future__ import annotations

import pytest

from evals.utils import (
    TrajectoryScorer,
    run_agent_async,
    EvalEnvironment
)

pytestmark = [
    pytest.mark.eval_category("outcome"),
]


# ============================================================
# A：Baseline outcome evals
# ============================================================

@pytest.mark.asyncio
@pytest.mark.eval_tier("tier_1")
@pytest.mark.eval_category("outcome")
async def test_detect_bigkey(agent):
    environment = EvalEnvironment(
        redis_data={
            # 普通 Key：作为背景噪声
            "user:1001": {"name": "alice", "level": "3"},
            "user:1002": {"name": "bob", "level": "5"},
            "config:app": "production",

            # 故意制造一个明显 BigKey：约 2 MB
            "cache:product:oversized": "x" * (2 * 1024 * 1024),

            # 再放一个稍大的 Key，避免环境过于单一
            "cache:homepage": "y" * (50 * 1024),
        },
    )

    query = """
    最近我的Redis内存占用有些异常，请检查并告诉我你发现了什么。
    """

    trajectory = await run_agent_async(
        agent=agent,
        query=query,
        model=model,
        environment=environment,
        session_id="eval-bigkey-001",
        max_iterations=10,
        scorer=(
            TrajectoryScorer()
            .success(
                llm_judge(
                    "Agent 必须识别出 `cache:product:oversized` 是当前 Redis 实例中明显的大 Key。",
                    "Agent 的结论必须建立在实际 Redis 诊断工具返回的证据上，执行轨迹中应包含针对 BigKey 的 Redis 检查。",
                    "Agent 应说明 `cache:product:oversized` 的大小明显高于环境中的其他 Key。",
                    "Agent 不应声称已经发生 OOM、Redis 内存泄漏、Key 淘汰或实例崩溃等没有证据支持的问题。",
                    include_tool_calls=True,
                )
            )
        ),
        eval_metadata={
            "category": "diagnosis",
            "scenario": "bigkey",
            "case_id": "bigkey_001",
        },
    )


# ============================================================
# B：Hillclimb outcome evals
# ============================================================
