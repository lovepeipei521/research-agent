import json
from pathlib import Path

from poirot.backend.evaluation.agentic_rag_benchmark import (
    AgentCase,
    _event_metrics,
    evaluate_agent_run,
    read_event_records,
    read_cases,
)


def test_read_agent_cases() -> None:
    cases = read_cases(Path("resource/poirot_agentic_rag_eval_v1.json"))

    assert len(cases) == 20
    assert cases[8].allow_query_refinement is True
    assert cases[10].should_call_knowledge_search is False


def test_agent_evaluation_checks_tool_source_and_contract() -> None:
    case = AgentCase(
        case_id="agent-test",
        prompt="查资料",
        should_call_knowledge_search=True,
        expected_sources=("sandbox.md",),
        required_strings=("Docker Sandbox",),
        required_any_groups=(),
        must_not_contain=(),
        min_knowledge_search_calls=1,
        allow_query_refinement=False,
    )
    records = [
        {"event_type": "tool.called", "payload": {"tool_name": "knowledge_search", "tool_input": {"query": "sandbox"}}},
        {"event_type": "tool.finished", "payload": {"tool_name": "knowledge_search", "output": json.dumps({"results": [{"path": "sandbox.md"}]})}},
    ]

    result = evaluate_agent_run("Docker Sandbox isolates commands", records, case)

    assert result["passed"] is True
    assert result["tool_selection_pass"] is True
    assert result["evidence_retrieved"] is True


def test_agent_evaluation_reads_evidence_middleware_command_wrapper() -> None:
    case = AgentCase(
        case_id="agent-wrapper",
        prompt="查资料",
        should_call_knowledge_search=True,
        expected_sources=("sandbox.md",),
        required_strings=("Docker Sandbox",),
        required_any_groups=(),
        must_not_contain=(),
        min_knowledge_search_calls=1,
        allow_query_refinement=False,
    )
    records = [
        {"event_type": "tool.called", "payload": {"tool_name": "knowledge_search", "tool_input": {"query": "sandbox"}}},
        {"event_type": "tool.finished", "payload": {"tool_name": "knowledge_search", "output": "Command(update={'observations': [{'content': '{\"results\": [{\"path\": \"sandbox.md\"}]}'}]})"}},
    ]

    result = evaluate_agent_run("Docker Sandbox isolates commands", records, case)

    assert result["evidence_retrieved"] is True
    assert result["passed"] is True


def test_agent_evaluation_detects_unnecessary_tool_call() -> None:
    case = AgentCase(
        case_id="agent-no-tool",
        prompt="打招呼",
        should_call_knowledge_search=False,
        expected_sources=(),
        required_strings=(),
        required_any_groups=(),
        must_not_contain=(),
        min_knowledge_search_calls=0,
        allow_query_refinement=False,
    )
    records = [{"event_type": "tool.called", "payload": {"tool_name": "knowledge_search"}}]

    result = evaluate_agent_run("你好", records, case)

    assert result["passed"] is False
    assert result["tool_selection_pass"] is False


def test_read_event_records_handles_pretty_json_objects(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps({"event_type": "tool.called", "payload": {}}, indent=2)
        + "\n\n"
        + json.dumps({"event_type": "tool.finished", "payload": {}}, indent=2),
        encoding="utf-8",
    )

    assert len(read_event_records(path)) == 2


def test_event_metrics_counts_tools_and_optional_token_usage() -> None:
    records = [
        {"event_type": "llm.request", "payload": {}},
        {
            "event_type": "llm.response",
            "payload": {"usage_metadata": {"input_tokens": 10, "output_tokens": 4}},
        },
        {"event_type": "tool.called", "payload": {"tool_name": "knowledge_search"}},
        {"event_type": "tool.finished", "payload": {"tool_name": "knowledge_search"}},
    ]

    assert _event_metrics(records) == {
        "llm_requests": 1,
        "tool_calls": 1,
        "tool_events": 2,
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }
