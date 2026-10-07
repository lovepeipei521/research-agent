import json

from poirot.backend.evaluation.resume_benchmark import _read_event_counts, evaluate_report


def test_evaluate_report_accepts_all_required_facts() -> None:
    passed, missing = evaluate_report("A 2026-08-01 B", ["A", "2026-08-01"])

    assert passed is True
    assert missing == []


def test_evaluate_report_surfaces_missing_facts() -> None:
    passed, missing = evaluate_report("A", ["A", "B"])

    assert passed is False
    assert missing == ["B"]


def test_evaluate_report_accepts_a_contract_alternative() -> None:
    passed, missing = evaluate_report(
        "现有资料无法确认。",
        ["资料"],
        [["不确定", "无法确认"]],
    )

    assert passed is True
    assert missing == []


def test_read_event_counts_supports_pretty_printed_jsonl_events(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps({"event_type": "llm.request"}, indent=2)
        + "\n\n"
        + json.dumps({"event_type": "tool.started"}, indent=2),
        encoding="utf-8",
    )

    assert _read_event_counts(events_path) == {"llm_requests": 1, "tool_events": 1}
