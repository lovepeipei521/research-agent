"""Controlled mechanism evaluation for DefaultStrategy context governance."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from poirot.backend.agents.context_engineering.contract import GovernanceContext
from poirot.backend.agents.context_engineering.strategies.default.strategy import DefaultStrategy
from poirot.backend.agents.context_engineering.utilities import token_counter
from poirot.backend.agents.middlewares.tagged_context_middleware import (
    POIROT_EXTERNALIZED_PATH,
    POIROT_SUMMARY,
)

_PROJECT_ROOT = Path(__file__).parents[3]
_DEFAULT_CASES = _PROJECT_ROOT / "resource" / "poirot_context_governance_eval_v1.json"


@dataclass(frozen=True)
class GovernanceCase:
    case_id: str
    kind: str
    data: dict[str, Any]


def read_cases(path: Path) -> list[GovernanceCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = raw.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("governance case file must contain a non-empty cases list")
    parsed: list[GovernanceCase] = []
    for item in cases:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("kind"), str):
            raise ValueError("each governance case requires id and kind")
        parsed.append(GovernanceCase(item["id"], item["kind"], item))
    return parsed


def _ctx(messages: list, governance: dict[str, Any], *, state: dict[str, Any] | None = None, counter=token_counter) -> GovernanceContext:
    return GovernanceContext(
        state=state or {},
        governance=governance,
        config={"window": 10_000},
        token_counter=counter,
        runtime=None,
        hook="benchmark",
        messages=messages,
    )


def _long_text(case_id: str, turn: int, tool: int, chars: int) -> str:
    anchor = f"ANCHOR-{case_id}-T{turn}-K{tool}"
    prefix = f"{anchor}. Controlled research evidence for externalization. "
    return prefix + ("evidence " * max(1, (chars - len(prefix)) // 9))


def _externalize_case(case: GovernanceCase, output_dir: Path) -> dict[str, Any]:
    turns = int(case.data["turns"])
    tools_per_turn = int(case.data["tools_per_turn"])
    content_chars = int(case.data["content_chars"])
    exempt_rounds = int(case.data["exempt_rounds"])
    messages: list = []
    for turn in range(turns):
        messages.append(HumanMessage(id=f"{case.case_id}-user-{turn}", content=f"Research turn {turn}"))
        for tool in range(tools_per_turn):
            messages.append(ToolMessage(
                id=f"{case.case_id}-tool-{turn}-{tool}",
                tool_call_id=f"{case.case_id}-tc-{turn}-{tool}",
                name="web_search",
                content=_long_text(case.case_id, turn, tool, content_chars),
            ))

    strategy = DefaultStrategy(params={
        "externalize_dir": str(output_dir / case.case_id / "externalized"),
        "externalize_min_chars": 400,
        "externalize_preview_chars": 240,
        "exempt_rounds": exempt_rounds,
    })
    before_tokens = token_counter(messages)
    governance = {"default": {"budget": {"fraction": 0.45}, "pending": ["P1"]}}
    result = strategy.before_model(_ctx(messages, governance))
    patches = [message for message in (result.messages_patch or []) if isinstance(message, ToolMessage)]
    replacements = {message.id: message for message in patches}
    rewritten = [replacements.get(message.id, message) for message in messages]
    after_tokens = token_counter(rewritten)
    paths = [Path(str(message.additional_kwargs.get(POIROT_EXTERNALIZED_PATH, ""))) for message in patches]
    integrity = all(path.is_file() and f"ANCHOR-{case.case_id}" in path.read_text(encoding="utf-8") for path in paths)
    reduction = max(0.0, (before_tokens - after_tokens) / before_tokens * 100) if before_tokens else 0.0
    expected_count = int(case.data["min_externalized"])
    min_reduction = float(case.data["min_token_reduction_percent"])
    passed = len(patches) >= expected_count and integrity and reduction >= min_reduction
    return {
        "id": case.case_id,
        "kind": case.kind,
        "passed": passed,
        "externalized_count": len(patches),
        "expected_min_externalized": expected_count,
        "externalization_integrity": integrity,
        "context_tokens_before": before_tokens,
        "context_tokens_after": after_tokens,
        "token_reduction_percent": round(reduction, 2),
        "required_min_reduction_percent": min_reduction,
        "externalized_paths": [str(path) for path in paths],
    }


class _DeterministicSummaryModel:
    """Exercises the P4 integration path without claiming semantic LLM quality."""

    def __init__(self, case_id: str) -> None:
        self._case_id = case_id

    def invoke(self, prompt: Any, config: Any = None) -> SimpleNamespace:
        return SimpleNamespace(text=f"Controlled compaction summary for {self._case_id}.")


def _summarize_case(case: GovernanceCase, output_dir: Path) -> dict[str, Any]:
    turns = int(case.data["turns"])
    preserve_recent = int(case.data["preserve_recent"])
    messages: list = []
    for turn in range(turns):
        messages.append(HumanMessage(id=f"{case.case_id}-user-{turn}", content=f"Historical fact {case.case_id}-{turn}"))
        messages.append(AIMessage(id=f"{case.case_id}-assistant-{turn}", content=f"Research response {case.case_id}-{turn}"))
    strategy = DefaultStrategy(
        params={
            "preserve_recent": preserve_recent,
            "externalize_dir": str(output_dir / case.case_id / "externalized"),
            "snapshot_dir": str(output_dir / case.case_id / "snapshots"),
        },
        model=_DeterministicSummaryModel(case.case_id),
    )
    governance = {"default": {"budget": {"fraction": 0.85}, "pending": ["P4"]}}
    result = strategy.before_model(_ctx(messages, governance, state={"research_question": "controlled test"}))
    patches = result.messages_patch or []
    summary_messages = [message for message in patches if isinstance(message, HumanMessage) and message.additional_kwargs.get(POIROT_SUMMARY)]
    preserved = [message for message in patches if not isinstance(message, RemoveMessage) and message not in summary_messages]
    recent_ids = {message.id for message in messages[-preserve_recent:]}
    preserved_ids = {message.id for message in preserved}
    summary_ok = len(summary_messages) == 1 and case.case_id in str(summary_messages[0].content)
    pass_preserve = preserved_ids == recent_ids
    return {
        "id": case.case_id,
        "kind": case.kind,
        "passed": summary_ok and pass_preserve,
        "summary_contract_pass": summary_ok,
        "recent_message_preservation_pass": pass_preserve,
        "messages_before": len(messages),
        "messages_retained": len(preserved),
        "messages_compacted": len(messages) - len(preserved),
        "note": "Uses a deterministic summary model; this verifies P4 wiring and recent-message retention, not semantic summary quality.",
    }


def _stop_case(case: GovernanceCase) -> dict[str, Any]:
    fraction = float(case.data["fraction"])
    window = 10_000
    messages = [
        HumanMessage(content="Continue researching"),
        AIMessage(
            id=f"{case.case_id}-assistant",
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "more"}, "id": f"{case.case_id}-tc", "type": "tool_call"}],
            additional_kwargs={"tool_calls": [{"name": "web_search", "id": f"{case.case_id}-tc"}]},
        ),
    ]
    strategy = DefaultStrategy()
    governance = {"default": {"budget": {"fraction": 0.0}, "seen_msgs": {}}}
    result = strategy.after_model(_ctx(messages, governance, counter=lambda _: int(window * fraction)))
    patches = result.messages_patch or []
    cleared = next((message for message in patches if isinstance(message, AIMessage) and message.id == f"{case.case_id}-assistant"), None)
    stop = next((message for message in patches if isinstance(message, HumanMessage) and message.name == "context_budget_stop"), None)
    tool_calls_removed = cleared is not None and not cleared.tool_calls and "tool_calls" not in (cleared.additional_kwargs or {})
    stop_notice = stop is not None and ("99%" in str(stop.content) if fraction >= 0.99 else "90%" in str(stop.content))
    return {
        "id": case.case_id,
        "kind": case.kind,
        "passed": result.jump_to == "model" and tool_calls_removed and stop_notice,
        "fraction": fraction,
        "jump_to_model": result.jump_to == "model",
        "tool_calls_removed": tool_calls_removed,
        "stop_notice_pass": stop_notice,
    }


def run_benchmark(cases: Sequence[GovernanceCase], *, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for case in cases:
        if case.kind == "externalize":
            results.append(_externalize_case(case, output_dir))
        elif case.kind == "summarize":
            results.append(_summarize_case(case, output_dir))
        elif case.kind == "stop_tool_calls":
            results.append(_stop_case(case))
        else:
            raise ValueError(f"unsupported governance case kind: {case.kind}")
    externalize_cases = [item for item in results if item["kind"] == "externalize"]
    summarize_cases = [item for item in results if item["kind"] == "summarize"]
    stop_cases = [item for item in results if item["kind"] == "stop_tool_calls"]
    summary = {
        "schema_version": "1.0",
        "benchmark": "poirot-context-governance-v1",
        "scope": "Synthetic long-history mechanism evaluation for DefaultStrategy; not an open-domain answer-quality benchmark.",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "aggregate": {
            "executed_cases": len(results),
            "passed_cases": sum(item["passed"] for item in results),
            "mechanism_pass_rate": round(sum(item["passed"] for item in results) / len(results) * 100, 2) if results else 0.0,
            "externalization_integrity_rate": round(sum(item["externalization_integrity"] for item in externalize_cases) / len(externalize_cases) * 100, 2) if externalize_cases else None,
            "mean_p1_token_reduction_percent": round(sum(item["token_reduction_percent"] for item in externalize_cases) / len(externalize_cases), 2) if externalize_cases else None,
            "p4_contract_pass_rate": round(sum(item["passed"] for item in summarize_cases) / len(summarize_cases) * 100, 2) if summarize_cases else None,
            "p5_tool_stop_pass_rate": round(sum(item["passed"] for item in stop_cases) / len(stop_cases) * 100, 2) if stop_cases else None,
        },
        "cases": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run controlled context-governance mechanism evaluation.")
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / ".poirot" / "evaluations" / "context-governance-v1")
    args = parser.parse_args(argv)
    summary = run_benchmark(read_cases(args.cases), output_dir=args.output_dir)
    aggregate = summary["aggregate"]
    print(
        f"{aggregate['passed_cases']}/{aggregate['executed_cases']} passed | "
        f"P1 token reduction={aggregate['mean_p1_token_reduction_percent']}% | "
        f"P5 stop={aggregate['p5_tool_stop_pass_rate']}%"
    )
    print(f"summary: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
