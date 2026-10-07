"""Bounded Agent behavior evaluation for the local Agentic RAG tool."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

from poirot.backend.evaluation.resume_benchmark import evaluate_report

_PROJECT_ROOT = Path(__file__).parents[3]
_DEFAULT_CASES = _PROJECT_ROOT / "resource" / "poirot_agentic_rag_eval_v1.json"
_DEFAULT_CORPUS = _PROJECT_ROOT / "resource" / "rag_eval_corpus"

# ``python -m`` bypasses the CLI entry point, so load the project .env here.
load_dotenv(_PROJECT_ROOT / ".env")


def _format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value}%"


@dataclass(frozen=True)
class AgentCase:
    case_id: str
    prompt: str
    should_call_knowledge_search: bool
    expected_sources: tuple[str, ...]
    required_strings: tuple[str, ...]
    required_any_groups: tuple[tuple[str, ...], ...]
    must_not_contain: tuple[str, ...]
    min_knowledge_search_calls: int
    allow_query_refinement: bool
    max_llm_requests: int | None = None
    max_tool_calls: int | None = None
    category: str = ""


def read_cases(path: Path) -> list[AgentCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = data.get("cases", [])
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Agent case file must contain a non-empty 'cases' list")
    parsed: list[AgentCase] = []
    for item in raw_cases:
        case_id = item.get("id")
        prompt = item.get("prompt")
        if not isinstance(case_id, str) or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("every Agent case requires string fields 'id' and 'prompt'")
        groups = item.get("required_any_groups", [])
        if not isinstance(groups, list) or not all(isinstance(group, list) and group for group in groups):
            raise ValueError(f"case {case_id!r} has invalid 'required_any_groups'")
        min_calls = item.get("min_knowledge_search_calls", 1 if item.get("should_call_knowledge_search", True) else 0)
        if not isinstance(min_calls, int) or min_calls < 0:
            raise ValueError(f"case {case_id!r} has invalid 'min_knowledge_search_calls'")
        parsed.append(
            AgentCase(
                case_id=case_id,
                prompt=prompt,
                should_call_knowledge_search=bool(item.get("should_call_knowledge_search", True)),
                expected_sources=tuple(item.get("expected_sources", [])),
                required_strings=tuple(item.get("required_strings", [])),
                required_any_groups=tuple(tuple(group) for group in groups),
                must_not_contain=tuple(item.get("must_not_contain", [])),
                min_knowledge_search_calls=min_calls,
                allow_query_refinement=bool(item.get("allow_query_refinement", False)),
                max_llm_requests=item.get("max_llm_requests"),
                max_tool_calls=item.get("max_tool_calls"),
                category=str(item.get("category", "")),
            )
        )
    return parsed


def _event_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Extract lightweight performance counters from a retained event trace."""
    llm_requests = sum(record.get("event_type") == "llm.request" for record in records)
    tool_calls = sum(record.get("event_type") == "tool.called" for record in records)
    tool_events = sum(str(record.get("event_type", "")).startswith("tool.") for record in records)

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    usage_seen = False
    for record in records:
        if record.get("event_type") != "llm.response":
            continue
        payload = record.get("payload") or {}
        candidates = [
            payload.get("usage"),
            payload.get("usage_metadata"),
            (payload.get("response_metadata") or {}).get("token_usage"),
        ]
        usage = next((item for item in candidates if isinstance(item, dict)), None)
        if usage is None:
            continue
        usage_seen = True
        input_value = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_value = usage.get("output_tokens", usage.get("completion_tokens", 0))
        total_value = usage.get("total_tokens")
        input_tokens += int(input_value or 0)
        output_tokens += int(output_value or 0)
        total_tokens += int(total_value or 0)

    if usage_seen and not total_tokens:
        total_tokens = input_tokens + output_tokens
    return {
        "llm_requests": llm_requests,
        "tool_calls": tool_calls,
        "tool_events": tool_events,
        "input_tokens": input_tokens if usage_seen else None,
        "output_tokens": output_tokens if usage_seen else None,
        "total_tokens": total_tokens if usage_seen else None,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def read_event_records(path: Path) -> list[dict[str, Any]]:
    """Read pretty-printed JSON objects separated by whitespace."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    position = 0
    records: list[dict[str, Any]] = []
    while position < len(raw):
        while position < len(raw) and raw[position].isspace():
            position += 1
        if position >= len(raw):
            break
        try:
            record, position = decoder.raw_decode(raw, position)
        except json.JSONDecodeError:
            break
        if isinstance(record, dict):
            records.append(record)
    return records


def _knowledge_events(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("event_type") in {"tool.called", "tool.finished"}
        and (record.get("payload") or {}).get("tool_name") == "knowledge_search"
    ]


def _retrieved_sources(records: Sequence[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for record in records:
        payload = record.get("payload") or {}
        if record.get("event_type") != "tool.finished" or payload.get("tool_name") != "knowledge_search":
            continue
        output_text = str(payload.get("output", ""))
        try:
            output = json.loads(output_text)
        except json.JSONDecodeError:
            output = None
        if isinstance(output, dict):
            for result in output.get("results", []):
                if isinstance(result, dict) and result.get("path"):
                    paths.add(str(result["path"]).replace("\\", "/").lower())
        # EvidenceMiddleware serialises the tool result as
        # ``Command(update={... content='{"results": ...}'})``.  The
        # journal still contains source paths, so recover them without
        # depending on Python repr details or evaluating the string.
        for match in re.finditer(r"['\"]path['\"]\s*:\s*['\"]([^'\"]+)", output_text):
            paths.add(match.group(1).replace("\\", "/").lower())
    return paths


def evaluate_agent_run(
    final_report: str,
    records: Sequence[dict[str, Any]],
    case: AgentCase,
) -> dict[str, Any]:
    knowledge_events = _knowledge_events(records)
    knowledge_calls = sum(record.get("event_type") == "tool.called" for record in knowledge_events)
    tool_names = {
        str((record.get("payload") or {}).get("tool_name", ""))
        for record in records
        if record.get("event_type") == "tool.called"
    }
    tool_selection_pass = (
        "knowledge_search" in tool_names
        if case.should_call_knowledge_search
        else "knowledge_search" not in tool_names
    )
    contract_pass, missing = evaluate_report(
        final_report,
        case.required_strings,
        case.required_any_groups,
    )
    forbidden = [term for term in case.must_not_contain if term in final_report]
    retrieved = _retrieved_sources(records)
    expected = {source.replace("\\", "/").lower() for source in case.expected_sources}
    evidence_retrieved = not expected or expected.issubset(retrieved)
    query_inputs = [
        (record.get("payload") or {}).get("tool_input", {}).get("query")
        for record in knowledge_events
        if record.get("event_type") == "tool.called"
    ]
    query_inputs = [str(query) for query in query_inputs if query]
    refinement_pass = knowledge_calls >= case.min_knowledge_search_calls
    if case.allow_query_refinement and case.min_knowledge_search_calls >= 2:
        refinement_pass = refinement_pass and len(set(query_inputs)) >= 2
    passed = tool_selection_pass and contract_pass and not forbidden and evidence_retrieved and refinement_pass
    return {
        "id": case.case_id,
        "category": case.category,
        "passed": passed,
        "should_call_knowledge_search": case.should_call_knowledge_search,
        "expected_sources": list(case.expected_sources),
        "tool_selection_pass": tool_selection_pass,
        "answer_contract_pass": contract_pass,
        "missing_required": missing,
        "forbidden_claims": forbidden,
        "evidence_retrieved": evidence_retrieved,
        "retrieved_sources": sorted(retrieved),
        "knowledge_search_calls": knowledge_calls,
        "query_inputs": query_inputs,
        "query_refinement_pass": refinement_pass,
        "query_refinement_applicable": case.allow_query_refinement,
        "tool_names": sorted(tool_names),
    }


def run_benchmark(
    cases: Sequence[AgentCase],
    *,
    provider: str,
    knowledge_dir: Path,
    output_dir: Path,
    benchmark_name: str = "poirot-agentic-rag-v1",
) -> dict[str, Any]:
    """Run Agent cases against one pinned provider and retain event evidence."""
    from poirot.backend.agents.memory.bootstrap import shutdown_memory_worker
    from poirot.backend.app.bootstrap import bootstrap_runtime

    output_dir.mkdir(parents=True, exist_ok=True)
    old_root = os.environ.get("POIROT_RAG_DIR")
    os.environ["POIROT_RAG_DIR"] = str(knowledge_dir)
    runtime = bootstrap_runtime(
        expert_mode=False,
        provider=provider,
        cli_overrides={"logs_root": str(output_dir / "runtime")},
    )
    results: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases, start=1):
            started = time.perf_counter()
            run_id = f"agentic-rag-{index:02d}-{case.case_id}"
            thread_id = f"agentic-rag-{case.case_id}"
            try:
                result = runtime.run_question(case.prompt, thread_id=thread_id, run_id=run_id)
                records = read_event_records(Path(result.events_path))
                item = evaluate_agent_run(result.final_report, records, case)
                metrics = _event_metrics(records)
                budget_failures: list[str] = []
                if case.max_llm_requests is not None and metrics["llm_requests"] > case.max_llm_requests:
                    budget_failures.append("llm_requests")
                if case.max_tool_calls is not None and metrics["tool_calls"] > case.max_tool_calls:
                    budget_failures.append("tool_calls")
                item.update({
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "trace": result.events_path,
                    "final_report": result.final_report,
                    **metrics,
                    "budget_pass": not budget_failures,
                    "budget_failures": budget_failures,
                })
            except Exception as exc:
                item = {
                    "id": case.case_id,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "llm_requests": 0,
                    "tool_calls": 0,
                    "tool_events": 0,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "budget_pass": False,
                }
            results.append(item)
    finally:
        shutdown_memory_worker()
        if old_root is None:
            os.environ.pop("POIROT_RAG_DIR", None)
        else:
            os.environ["POIROT_RAG_DIR"] = old_root

    scored = [item for item in results if "error" not in item]
    evidence_cases = [item for item in scored if item.get("expected_sources")]
    refinement_cases = [item for item in results if item.get("query_refinement_applicable")]
    latency_values = [float(item.get("elapsed_ms", 0)) for item in scored]
    token_cases = [item for item in scored if item.get("total_tokens") is not None]
    summary = {
        "schema_version": "1.0",
        "benchmark": benchmark_name,
        "scope": "Controlled Agent runs; quality, tool use, budgets, latency, and optional token usage",
        "configuration": {"provider": provider, "knowledge_dir": str(knowledge_dir), "case_count": len(cases)},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "aggregate": {
            "executed_cases": len(results),
            "scored_cases": len(scored),
            "runtime_error_cases": len(results) - len(scored),
            "passed_cases": sum(item.get("passed", False) for item in results),
            "case_completion_rate": round(sum(item.get("passed", False) for item in scored) / len(scored) * 100, 2) if scored else None,
            "tool_selection_rate": round(sum(item.get("tool_selection_pass", False) for item in scored) / len(scored) * 100, 2) if scored else None,
            "answer_contract_rate": round(sum(item.get("answer_contract_pass", False) for item in scored) / len(scored) * 100, 2) if scored else None,
            "evidence_retrieval_rate": round(sum(item.get("evidence_retrieved", False) for item in evidence_cases) / len(evidence_cases) * 100, 2) if evidence_cases else None,
            "query_refinement_rate": round(sum(item.get("query_refinement_pass", False) for item in refinement_cases) / len(refinement_cases) * 100, 2) if refinement_cases else 0.0,
            "mean_knowledge_search_calls": round(sum(item.get("knowledge_search_calls", 0) for item in scored) / len(scored), 2) if scored else None,
            "mean_llm_requests": round(sum(item.get("llm_requests", 0) for item in scored) / len(scored), 2) if scored else None,
            "mean_tool_calls": round(sum(item.get("tool_calls", 0) for item in scored) / len(scored), 2) if scored else None,
            "mean_tool_events": round(sum(item.get("tool_events", 0) for item in scored) / len(scored), 2) if scored else None,
            "budget_pass_rate": round(sum(item.get("budget_pass", False) for item in scored) / len(scored) * 100, 2) if scored else None,
            "mean_latency_ms": round(sum(latency_values) / len(latency_values), 2) if latency_values else None,
            "p50_latency_ms": round(_percentile(latency_values, 0.50), 2) if latency_values else None,
            "p95_latency_ms": round(_percentile(latency_values, 0.95), 2) if latency_values else None,
            "mean_input_tokens": round(sum(item["input_tokens"] for item in token_cases) / len(token_cases), 2) if token_cases else None,
            "mean_output_tokens": round(sum(item["output_tokens"] for item in token_cases) / len(token_cases), 2) if token_cases else None,
            "mean_total_tokens": round(sum(item["total_tokens"] for item in token_cases) / len(token_cases), 2) if token_cases else None,
            "token_usage_cases": len(token_cases),
        },
        "cases": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded Agentic RAG behavior benchmark.")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    parser.add_argument("--knowledge-dir", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / ".poirot" / "evaluations" / "agentic-rag-v1")
    parser.add_argument("--benchmark-name", default="poirot-agentic-rag-v1")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    cases = read_cases(args.cases)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        cases = cases[: args.limit]
    summary = run_benchmark(
        cases,
        provider=args.provider,
        knowledge_dir=args.knowledge_dir,
        output_dir=args.output_dir,
        benchmark_name=args.benchmark_name,
    )
    aggregate = summary["aggregate"]
    print(
        f"{aggregate['passed_cases']}/{aggregate['scored_cases']} scored passed "
        f"({aggregate['runtime_error_cases']} runtime errors) | "
        f"tool_selection={_format_percent(aggregate['tool_selection_rate'])} | "
        f"answer_contract={_format_percent(aggregate['answer_contract_rate'])}"
    )
    print(f"summary: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
