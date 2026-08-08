"""验证从 original 保留下来的消息输出格式契约。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from redis_sre_agent.agent import langgraph_agent as langgraph_agent_module
from redis_sre_agent.agent.chat_agent import CHAT_SYSTEM_PROMPT
from redis_sre_agent.agent.langgraph_agent import SRELangGraphAgent
from redis_sre_agent.agent.prompts import (
    REDIS_COMMAND_SEMANTICS_GUARDRAILS,
    SRE_SYSTEM_PROMPT,
)


_REQUIRED_HEADINGS = (
    "## Initial Assessment",
    "## What I'm Seeing",
    "## My Recommendation",
    "## Supporting Info",
)


class _CapturingComposerLLM:
    def __init__(self, content: Any) -> None:
        self.content = content
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.messages = list(messages)
        return SimpleNamespace(content=self.content)


def test_deep_triage_uses_shared_original_prompt() -> None:
    assert langgraph_agent_module.SRE_SYSTEM_PROMPT == SRE_SYSTEM_PROMPT
    positions = [SRE_SYSTEM_PROMPT.index(heading) for heading in _REQUIRED_HEADINGS]

    assert positions == sorted(positions)
    assert "## Formatting Requirements" in SRE_SYSTEM_PROMPT
    assert "According to [Document Title] (source: [source_path])" in SRE_SYSTEM_PROMPT


def test_chat_prompt_keeps_original_output_and_semantics_rules() -> None:
    assert REDIS_COMMAND_SEMANTICS_GUARDRAILS in CHAT_SYSTEM_PROMPT
    assert "copy those headings verbatim" in CHAT_SYSTEM_PROMPT
    assert "Return only the requested document or answer body" in CHAT_SYSTEM_PROMPT
    assert "Be conversational about what you're finding" in CHAT_SYSTEM_PROMPT


def test_deterministic_composer_fallback_keeps_required_sections() -> None:
    result = SRELangGraphAgent._build_final_markdown_fallback(
        initial_writeup="assessment",
        recommendations=[],
        topics=[],
    )
    positions = [result.index(heading) for heading in _REQUIRED_HEADINGS]

    assert positions == sorted(positions)
    assert all(result.count(heading) == 1 for heading in _REQUIRED_HEADINGS)


@pytest.mark.asyncio
async def test_composer_uses_exact_heading_order_and_markdown_only_contract() -> None:
    llm = _CapturingComposerLLM(
        [
            {
                "content": (
                    "## Initial Assessment\nassessment\n\n"
                    "## What I'm Seeing\nfinding\n\n"
                    "## My Recommendation\nrecommendation\n\n"
                    "## Supporting Info\nsource\n\n"
                    "## Safety and Fact Checking\nmodel-added section"
                )
            }
        ]
    )
    agent = SRELangGraphAgent(llm=llm)

    result = await agent._compose_final_markdown(
        initial_assessment_lines=["assessment"],
        per_topic_recommendations=[],
        instance_ctx={},
    )

    assert "## Safety and Fact Checking" not in result
    assert isinstance(llm.messages[0], SystemMessage)
    assert isinstance(llm.messages[1], HumanMessage)
    system_prompt = str(llm.messages[0].content)
    format_prompt = str(llm.messages[1].content)
    positions = [format_prompt.index(heading) for heading in _REQUIRED_HEADINGS]
    assert positions == sorted(positions)
    assert "Do NOT invent facts, commands, endpoints, or metrics" in system_prompt
    assert "Return Markdown only" in format_prompt


@pytest.mark.asyncio
async def test_composer_keeps_safety_section_only_when_notes_exist() -> None:
    llm = _CapturingComposerLLM(
        "## Initial Assessment\nassessment\n\n"
        "## What I'm Seeing\nfinding\n\n"
        "## My Recommendation\nrecommendation\n\n"
        "## Supporting Info\nsource\n\n"
        "## Safety and Fact Checking\n- verified"
    )
    agent = SRELangGraphAgent(llm=llm)

    result = await agent._compose_final_markdown(
        initial_assessment_lines=["assessment"],
        per_topic_recommendations=[],
        instance_ctx={},
        safety_and_fact_check_notes=[{"status": "verified"}],
    )

    assert "## Safety and Fact Checking\n- verified" in result
    assert '"safety_and_fact_check_notes"' in str(llm.messages[1].content)
