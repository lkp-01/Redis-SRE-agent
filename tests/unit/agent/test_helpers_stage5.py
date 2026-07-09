"""Stage 5 Agent helper 测试。"""

from __future__ import annotations

from redis_sre_agent.agent.helpers import build_result_envelope, coerce_response_text, extract_citations
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
                "data": {
                    "retrieval_kind": "dummy_knowledge",
                    "results": [{"source": "redis-docs", "snippet": "memory"}],
                },
            },
        ]
    )

    assert citations == [
        {
            "source": "redis-docs",
            "snippet": "memory",
            "retrieval_kind": "dummy_knowledge",
            "retrieval_label": "Knowledge search",
        }
    ]


def test_agent_response_derives_search_results_from_knowledge_envelopes() -> None:
    response = AgentResponse(
        response="ok",
        tool_envelopes=[
            {
                "tool_key": "knowledge_abcdef_search",
                "name": "search",
                "data": {
                    "retrieval_kind": "dummy_knowledge",
                    "results": [{"source": "stage5", "snippet": "slot"}],
                },
            }
        ],
    )

    assert response.search_results == [
        {
            "source": "stage5",
            "snippet": "slot",
            "retrieval_kind": "dummy_knowledge",
            "retrieval_label": "Knowledge search",
        }
    ]


def test_coerce_response_text_handles_list_content() -> None:
    assert coerce_response_text([{"text": " hello "}, "world"]) == "hello\nworld"
