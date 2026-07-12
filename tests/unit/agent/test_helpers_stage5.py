"""Stage 5 Agent helper 测试。"""

from __future__ import annotations

import pytest

from redis_sre_agent.agent.helpers import (
    build_adapters_for_tooldefs,
    build_result_envelope,
    coerce_response_text,
    extract_citations,
)
from redis_sre_agent.agent.models import AgentResponse
from redis_sre_agent.tools.models import ToolCapability, ToolDefinition


def test_build_result_envelope_preserves_full_data_and_description() -> None:
    tool = ToolDefinition(
        name="redis_command_abcdef_info",
        description="Get Redis INFO.",
        capability=ToolCapability.DIAGNOSTICS,
        parameters={"type": "object", "properties": {}},
    )
    result = {"status": "success", "data": {"redis_version": "7.2.0"}}

    envelope = build_result_envelope(tool.name, {"section": "memory"}, result, {tool.name: tool})

    assert envelope["tool_key"] == tool.name
    assert envelope["name"] == "info"
    assert envelope["description"] == "Get Redis INFO."
    assert envelope["args"] == {"section": "memory"}
    assert envelope["status"] == "success"
    assert envelope["data"] == result


def test_build_result_envelope_keeps_plain_compound_tool_names() -> None:
    for tool_name, expected in {
        "resolve_redis_targets": "resolve_redis_targets",
        "memory_stats": "memory_stats",
        "client_list": "client_list",
        "config_get": "config_get",
    }.items():
        envelope = build_result_envelope(
            tool_name,
            {},
            {"status": "success"},
            {},
        )

        assert envelope["name"] == expected


def test_build_result_envelope_marks_error_status_and_summarizes_large_data() -> None:
    tool_name = "redis_command_abcdef_slowlog"
    large_result = {"status": "failed", "entries": [{"command": "GET x"}] * 300}

    envelope = build_result_envelope(tool_name, {}, large_result, {})

    assert envelope["status"] == "error"
    assert envelope["data"] == large_result
    assert envelope["summary"]
    assert "truncated" in envelope["summary"]


def test_extract_citations_reads_knowledge_results_only() -> None:
    citations = extract_citations(
        [
            {
                "tool_key": "redis_command_abcdef_info",
                "name": "info",
                "data": {"results": [{"source": "wrong"}]},
            },
            {
                "tool_key": "knowledge_abcdef_search",
                "name": "search",
                "status": "success",
                "data": {
                    "status": "success",
                    "retrieval_kind": "knowledge_search",
                    "results": [
                        {
                            "title": "Memory runbook",
                            "source": "file://shared/memory.md",
                            "document_hash": "doc",
                            "chunk_index": 0,
                            "score": 0.1,
                            "content": "Inspect INFO memory.",
                        }
                    ],
                },
            },
        ]
    )

    assert citations == [
        {
            "title": "Memory runbook",
            "source": "file://shared/memory.md",
            "document_hash": "doc",
            "chunk_index": 0,
            "score": 0.1,
            "content": "Inspect INFO memory.",
            "retrieval_kind": "knowledge_search",
            "retrieval_label": "Knowledge search",
        }
    ]


def test_extract_citations_ignores_failed_or_sourceless_results() -> None:
    citations = extract_citations(
        [
            {
                "tool_key": "knowledge_abcdef_search",
                "name": "search",
                "status": "error",
                "data": {
                    "status": "failed",
                    "results": [{"source": "must-not-leak"}],
                },
            },
            {
                "tool_key": "knowledge_abcdef_search",
                "name": "search",
                "status": "success",
                "data": {
                    "status": "success",
                    "results": [{"title": "No authoritative source"}],
                },
            },
        ]
    )

    assert citations == []


def test_agent_response_derives_search_results_from_knowledge_envelopes() -> None:
    response = AgentResponse(
        response="ok",
        tool_envelopes=[
            {
                "tool_key": "knowledge_abcdef_search",
                "name": "search",
                "status": "success",
                "data": {
                    "status": "success",
                    "retrieval_kind": "knowledge_search",
                    "results": [
                        {
                            "title": "Latency runbook",
                            "source": "file://shared/latency.md",
                            "document_hash": "doc",
                            "chunk_index": 1,
                            "score": 0.2,
                        }
                    ],
                },
            }
        ],
    )

    assert response.search_results == [
        {
            "title": "Latency runbook",
            "source": "file://shared/latency.md",
            "document_hash": "doc",
            "chunk_index": 1,
            "score": 0.2,
            "retrieval_kind": "knowledge_search",
            "retrieval_label": "Knowledge search",
        }
    ]


def test_coerce_response_text_handles_list_content() -> None:
    assert coerce_response_text([{"text": " hello "}, "world"]) == "hello\nworld"


@pytest.mark.asyncio
async def test_mcp_json_schema_round_trips_through_langchain_adapter() -> None:
    calls: list[tuple[str, dict]] = []

    class RecordingManager:
        async def resolve_tool_call(self, name: str, args: dict):
            calls.append((name, dict(args)))
            return {"status": "success", "echo": dict(args)}

    tool = ToolDefinition(
        name="mcp_schema_a1b2c3_read_status",
        description="Read external status.",
        capability=ToolCapability.DIAGNOSTICS,
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "enabled": {"type": "boolean"},
                "note": {"type": ["string", "null"]},
                "opaque": {"description": "Optional provider-defined value."},
            },
            "required": ["text", "count", "ratio", "enabled"],
        },
    )

    adapters = await build_adapters_for_tooldefs(RecordingManager(), [tool])
    assert len(adapters) == 1
    adapter = adapters[0]
    schema = adapter.args_schema.model_json_schema()
    values = {
        "text": "ok",
        "count": 2,
        "ratio": 0.5,
        "enabled": True,
        "note": None,
        "opaque": {"provider": "value"},
    }

    result = await adapter.ainvoke(values)

    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["count"]["type"] == "integer"
    assert schema["properties"]["ratio"]["type"] == "number"
    assert schema["properties"]["enabled"]["type"] == "boolean"
    assert {item.get("type") for item in schema["properties"]["note"]["anyOf"]} == {
        "string",
        "null",
    }
    assert result == {"status": "success", "echo": values}
    assert calls == [(tool.name, values)]
