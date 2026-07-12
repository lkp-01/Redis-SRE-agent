"""Recommendation worker 内部 knowledge ToolMessage 的 evidence 回传测试。"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from redis_sre_agent.agent.helpers import build_adapters_for_tooldefs, extract_citations
from redis_sre_agent.agent.models import Recommendation, RecommendationStep
from redis_sre_agent.agent.subgraphs.recommendation_worker import (
    build_recommendation_worker,
)
from redis_sre_agent.tools.models import ToolCapability, ToolDefinition


class FakeManager:
    def __init__(self, *, failed: bool = False) -> None:
        self.failed = failed
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def resolve_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(args)))
        if self.failed:
            return {"status": "failed", "error": "safe simulated failure"}
        index = len(self.calls) - 1
        return {
            "status": "success",
            "retrieval_kind": "knowledge_search",
            "retrieval_label": "Knowledge search",
            "results": [
                {
                    "title": f"Runbook {index}",
                    "source": f"file://shared/runbook-{index}.md",
                    "document_hash": f"doc-{index}",
                    "chunk_index": index,
                    "score": 0.1 + index,
                    "content": "Evidence-backed guidance.",
                }
            ],
        }


class FakeStructuredRecommendation:
    async def ainvoke(self, _messages: list[Any]) -> Recommendation:
        return Recommendation(
            topic_id="T1",
            title="Memory pressure",
            steps=[RecommendationStep(description="Use knowledge evidence")],
        )


class WorkerKnowledgeLLM:
    def __init__(self, *, rounds: int) -> None:
        self.rounds = rounds
        self.tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "WorkerKnowledgeLLM":
        self.tools = list(tools)
        return self

    def with_structured_output(
        self,
        schema: type[Any],
        **_kwargs: Any,
    ) -> FakeStructuredRecommendation:
        assert schema is Recommendation
        return FakeStructuredRecommendation()

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        completed = sum(isinstance(message, ToolMessage) for message in messages)
        if completed < self.rounds:
            tool = next(
                item
                for item in self.tools
                if item.name.startswith("knowledge_") and item.name.endswith("search")
            )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"knowledge-call-{completed}",
                        "name": tool.name,
                        "args": {"query": f"memory pressure {completed}"},
                    }
                ],
            )
        return AIMessage(content="Knowledge collection complete", tool_calls=[])


def _definition() -> ToolDefinition:
    return ToolDefinition(
        name="knowledge_abcdef_search",
        description="搜索 knowledge base。",
        capability=ToolCapability.KNOWLEDGE,
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )


async def _run_worker(*, rounds: int, failed: bool = False) -> tuple[Any, FakeManager]:
    definition = _definition()
    manager = FakeManager(failed=failed)
    adapters = await build_adapters_for_tooldefs(manager, [definition])
    worker = build_recommendation_worker(
        WorkerKnowledgeLLM(rounds=rounds),
        adapters,
        knowledge_tooldefs_by_name={definition.name: definition},
        max_tool_steps=3,
    )
    state = await worker.ainvoke(
        {
            "messages": [],
            "budget": 3,
            "topic": {"id": "T1", "title": "Memory pressure"},
            "evidence": [],
            "instance": {},
            "knowledge_envelopes": [],
        }
    )
    return state, manager


@pytest.mark.asyncio
async def test_worker_converts_tool_message_without_reexecuting_tool() -> None:
    state, manager = await _run_worker(rounds=1)

    assert len(manager.calls) == 1
    assert len(state["knowledge_envelopes"]) == 1
    envelope = state["knowledge_envelopes"][0]
    assert envelope["tool_key"] == "knowledge_abcdef_search"
    assert envelope["args"] == {"query": "memory pressure 0"}
    assert envelope["status"] == "success"
    assert envelope["data"]["results"][0]["document_hash"] == "doc-0"
    assert state["result"]["topic_id"] == "T1"


@pytest.mark.asyncio
async def test_worker_accumulates_multiple_knowledge_rounds() -> None:
    state, manager = await _run_worker(rounds=2)

    assert len(manager.calls) == 2
    assert len(state["knowledge_envelopes"]) == 2
    assert [item["data"]["results"][0]["document_hash"] for item in state["knowledge_envelopes"]] == [
        "doc-0",
        "doc-1",
    ]


@pytest.mark.asyncio
async def test_worker_retains_failed_envelope_but_does_not_create_citation() -> None:
    state, manager = await _run_worker(rounds=1, failed=True)

    assert len(manager.calls) == 1
    assert state["knowledge_envelopes"][0]["status"] == "error"
    assert extract_citations(state["knowledge_envelopes"]) == []
