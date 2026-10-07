"""Run bounded end-to-end benchmarks and retain the evidence for resume metrics.

The runner intentionally evaluates only the capability described by its case
file.  It does not turn a controlled source-grounded benchmark into a claim
about general web-research accuracy or online multi-agent collaboration.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from poirot.backend.app.bootstrap import bootstrap_runtime
from poirot.backend.agents.memory.bootstrap import shutdown_memory_worker

_PROJECT_ROOT = Path(__file__).parents[3]
_DEFAULT_CASES = _PROJECT_ROOT / "resource" / "poirot_resume_e2e_v1.json"


@dataclass(frozen=True)
class BenchmarkCase:
    """One deterministic acceptance contract for an end-to-end run."""

    case_id: str
    prompt: str
    required_strings: tuple[str, ...]
    required_any_groups: tuple[tuple[str, ...], ...] = ()


def _read_cases(path: Path) -> list[BenchmarkCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark case file must contain a non-empty 'cases' list")

    parsed: list[BenchmarkCase] = []
    for item in cases:
        case_id = item.get("id")
        prompt = item.get("prompt")
        expected = item.get("required_strings")
        alternatives = item.get("required_any_groups", [])
        if not isinstance(case_id, str) or not isinstance(prompt, str):
            raise ValueError("every case requires string fields 'id' and 'prompt'")
        if not isinstance(expected, list) or not expected or not all(isinstance(v, str) for v in expected):
            raise ValueError(f"case {case_id!r} requires a non-empty string 'required_strings' list")
        if (
            not isinstance(alternatives, list)
            or not all(
                isinstance(group, list)
                and group
                and all(isinstance(value, str) for value in group)
                for group in alternatives
            )
        ):
            raise ValueError(f"case {case_id!r} has invalid 'required_any_groups'")
        parsed.append(
            BenchmarkCase(
                case_id,
                prompt,
                tuple(expected),
                tuple(tuple(group) for group in alternatives),
            )
        )
    return parsed


def evaluate_report(
    report: str,
    required_strings: Sequence[str],
    required_any_groups: Sequence[Sequence[str]] = (),
) -> tuple[bool, list[str]]:
    """Return a transparent literal-contract verdict for a final response."""
    missing = [value for value in required_strings if value not in report]
    for alternatives in required_any_groups:
        if not any(value in report for value in alternatives):
            missing.append("one_of(" + " | ".join(alternatives) + ")")
    return not missing, missing


def _read_event_counts(path: Path) -> dict[str, int]:
    counts = {"llm_requests": 0, "tool_events": 0}
    if not path.exists():
        return counts
    raw_events = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    position = 0
    while position < len(raw_events):
        while position < len(raw_events) and raw_events[position].isspace():
            position += 1
        if position >= len(raw_events):
            break
        try:
            event, position = decoder.raw_decode(raw_events, position)
        except json.JSONDecodeError:
            break
        event_type = str(event.get("event_type", ""))
        if event_type == "llm.request":
            counts["llm_requests"] += 1
        if event_type.startswith("tool."):
            counts["tool_events"] += 1
    return counts


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def run_benchmark(
    cases: Sequence[BenchmarkCase],
    *,
    provider: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute cases against one pinned provider and write raw run evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = bootstrap_runtime(
        expert_mode=False,
        provider=provider,
        cli_overrides={"logs_root": str(output_dir / "runtime")},
    )
    results: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        for index, case in enumerate(cases, start=1):
            run_id = f"resume-e2e-{index:02d}-{case.case_id}"
            thread_id = f"resume-e2e-{case.case_id}"
            started = time.perf_counter()
            try:
                result = runtime.run_question(
                    case.prompt,
                    thread_id=thread_id,
                    run_id=run_id,
                )
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                passed, missing = evaluate_report(
                    result.final_report,
                    case.required_strings,
                    case.required_any_groups,
                )
                counts = _read_event_counts(Path(result.events_path))
                results.append(
                    {
                        "id": case.case_id,
                        "passed": passed,
                        "missing_required_strings": missing,
                        "elapsed_ms": elapsed_ms,
                        "llm_requests": counts["llm_requests"],
                        "tool_events": counts["tool_events"],
                        "trace": result.events_path,
                        "final_report": result.final_report,
                    }
                )
            except Exception as exc:  # The exception is evaluation evidence, not a runner crash.
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                results.append(
                    {
                        "id": case.case_id,
                        "passed": False,
                        "missing_required_strings": list(case.required_strings),
                        "elapsed_ms": elapsed_ms,
                        "llm_requests": 0,
                        "tool_events": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        shutdown_memory_worker()

    completed = len(results)
    passed_count = sum(item["passed"] for item in results)
    latency_values = [float(item["elapsed_ms"]) for item in results]
    report = {
        "schema_version": "1.0",
        "benchmark": "poirot-resume-e2e-v1",
        "scope": "20 controlled source-grounded Agent tasks; expert mode disabled",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "provider": provider,
            "expert_mode": False,
            "case_count": len(cases),
        },
        "aggregate": {
            "executed_cases": completed,
            "passed_cases": passed_count,
            "task_completion_rate": round(passed_count / completed * 100, 2) if completed else 0.0,
            "mean_latency_ms": round(sum(latency_values) / completed, 2) if completed else 0.0,
            "p50_latency_ms": round(_percentile(latency_values, 0.50), 2),
            "p95_latency_ms": round(_percentile(latency_values, 0.95), 2),
            "mean_llm_requests": round(sum(item["llm_requests"] for item in results) / completed, 2) if completed else 0.0,
            "mean_tool_events": round(sum(item["tool_events"] for item in results) / completed, 2) if completed else 0.0,
        },
        "cases": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def rescore_summary(
    source_summary: Path,
    cases: Sequence[BenchmarkCase],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Re-score retained predictions after an acceptance-contract correction."""
    source = json.loads(source_summary.read_text(encoding="utf-8"))
    case_map = {case.case_id: case for case in cases}
    results = list(source.get("cases", []))
    if len(results) != len(cases):
        raise ValueError("source summary case count does not match the supplied case file")

    for item in results:
        case = case_map.get(item.get("id"))
        if case is None or "final_report" not in item:
            raise ValueError("source summary is missing a scored final report")
        passed, missing = evaluate_report(
            str(item["final_report"]),
            case.required_strings,
            case.required_any_groups,
        )
        item["passed"] = passed
        item["missing_required_strings"] = missing

    completed = len(results)
    passed_count = sum(item["passed"] for item in results)
    latency_values = [float(item["elapsed_ms"]) for item in results]
    source["rescored_at"] = datetime.now(timezone.utc).isoformat()
    source["scoring_source"] = str(source_summary)
    source["aggregate"] = {
        "executed_cases": completed,
        "passed_cases": passed_count,
        "task_completion_rate": round(passed_count / completed * 100, 2) if completed else 0.0,
        "mean_latency_ms": round(sum(latency_values) / completed, 2) if completed else 0.0,
        "p50_latency_ms": round(_percentile(latency_values, 0.50), 2),
        "p95_latency_ms": round(_percentile(latency_values, 0.95), 2),
        "mean_llm_requests": round(sum(item["llm_requests"] for item in results) / completed, 2) if completed else 0.0,
        "mean_tool_events": round(sum(item["tool_events"] for item in results) / completed, 2) if completed else 0.0,
    }
    source["cases"] = results
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return source


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded Poirot resume benchmark.")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / ".poirot" / "evaluations" / "resume-e2e-v1")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    parser.add_argument("--rescore-summary", type=Path, default=None, help="Re-score retained final reports without model calls.")
    args = parser.parse_args(argv)

    cases = _read_cases(args.cases)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be greater than zero")
        cases = cases[: args.limit]
    if args.rescore_summary:
        report = rescore_summary(args.rescore_summary, cases, output_dir=args.output_dir)
    else:
        report = run_benchmark(cases, provider=args.provider, output_dir=args.output_dir)
    aggregate = report["aggregate"]
    print(
        f"{aggregate['passed_cases']}/{aggregate['executed_cases']} passed | "
        f"completion={aggregate['task_completion_rate']}% | "
        f"p95={aggregate['p95_latency_ms']}ms"
    )
    print(f"summary: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
