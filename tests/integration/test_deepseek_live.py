"""真实 DeepSeek API smoke tests。

只有设置 `RUN_DEEPSEEK_LIVE_TESTS=1` 且本地存在 API key 时才运行。测试不会打印
key、完整响应或请求头；它只验证文本响应和一次无副作用工具调用契约。
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from redis_sre_agent.core.llm_helpers import create_llm, has_llm_credentials


_LIVE_ENABLED = os.getenv("RUN_DEEPSEEK_LIVE_TESTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_LIVE_ENABLED and has_llm_credentials()),
        reason="需要 RUN_DEEPSEEK_LIVE_TESTS=1 和本地 DeepSeek API key",
    ),
]


@tool
def local_health_probe(component: str) -> str:
    """返回本地测试组件的固定健康状态，不访问 Redis 或其他外部服务。"""

    return f"{component}:healthy"


@pytest.mark.asyncio
async def test_deepseek_live_text_response() -> None:
    model = create_llm()

    response = await model.ainvoke([HumanMessage(content="只回复 READY")])

    assert str(response.content).strip()


@pytest.mark.asyncio
async def test_deepseek_live_tool_call_round_trip() -> None:
    model = create_llm()
    model_with_tools = model.bind_tools([local_health_probe])
    messages = [
        HumanMessage(
            content=(
                "必须调用 local_health_probe，参数 component 使用 redis；"
                "拿到工具结果后再给一句简短结论。"
            )
        )
    ]

    tool_request = await model_with_tools.ainvoke(messages)

    assert tool_request.tool_calls
    call = tool_request.tool_calls[0]
    assert call["name"] == "local_health_probe"
    tool_result = local_health_probe.invoke(call["args"])

    final_response = await model_with_tools.ainvoke(
        [
            *messages,
            tool_request,
            ToolMessage(content=tool_result, tool_call_id=call["id"]),
        ]
    )

    assert str(final_response.content).strip()
