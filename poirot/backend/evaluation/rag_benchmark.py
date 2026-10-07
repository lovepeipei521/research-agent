"""Bounded retrieval evaluation for the lightweight local Agentic RAG."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from poirot.backend.agents.agent_tools.builtin.knowledge_search import (
    knowledge_search,
    reset_knowledge_index,
)

_PROJECT_ROOT = Path(__file__).parents[3]
_DEFAULT_CASES = _PROJECT_ROOT / "resource" / "poirot_rag_eval_v1.json"
_DEFAULT_CORPUS = _PROJECT_ROOT / "resource" / "rag_eval_corpus"


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    query: str
    expected_sources: tuple[str, ...]
    required_terms: tuple[str, ...]
    should_find: bool


def read_cases(path: Path) -> list[RetrievalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = data.get("cases", [])
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("retrieval case file must contain a non-empty 'cases' list")

    cases: list[RetrievalCase] = []
    for item in raw_cases:
        case_id = item.get("id")
        query = item.get("query")
        expected_sources = item.get("expected_sources", [])
        required_terms = item.get("required_terms", [])
        should_find = item.get("should_find", True)
        if not isinstance(case_id, str) or not isinstance(query, str) or not query.strip():
            raise ValueError("every retrieval case requires string fields 'id' and 'query'")
        if not isinstance(expected_sources, list) or not all(isinstance(v, str) for v in expected_sources):
            raise ValueError(f"case {case_id!r} has invalid 'expected_sources'")
        if not isinstance(required_terms, list) or not all(isinstance(v, str) for v in required_terms):
            raise ValueError(f"case {case_id!r} has invalid 'required_terms'")
        if not isinstance(should_find, bool):
            raise ValueError(f"case {case_id!r} has invalid 'should_find'")
        cases.append(
            RetrievalCase(
                case_id=case_id,
                query=query,
                expected_sources=tuple(expected_sources),
                required_terms=tuple(required_terms),
                should_find=should_find,
            )
        )
    return cases


def _normalise_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./").lower()


def evaluate_results(
    result: dict[str, Any],
    case: RetrievalCase,
    *,
    top_k: int,
) -> dict[str, Any]:
    results = result.get("results", [])
    if not isinstance(results, list):
        results = []
    results = results[:top_k]
    paths = {_normalise_path(str(item.get("path", ""))) for item in results if isinstance(item, dict)}
    snippets = "\n".join(
        f"{item.get('title', '')}\n{item.get('snippet', '')}"
        for item in results
        if isinstance(item, dict)
    )
    # Chunk excerpts can contain line wrapping. Required facts are textual,
    # not layout-sensitive, so normalise whitespace before matching phrases.
    snippets = re.sub(r"\s+", " ", snippets).lower()

    source_hit = bool(case.expected_sources) and all(
        _normalise_path(source) in paths for source in case.expected_sources
    )
    found_terms = [term for term in case.required_terms if term.lower() in snippets]
    term_coverage = (
        len(found_terms) / len(case.required_terms) if case.required_terms else (1.0 if source_hit else 0.0)
    )
    if case.should_find:
        passed = source_hit and term_coverage == 1.0
    else:
        passed = not results

    return {
        "id": case.case_id,
        "passed": passed,
        "source_hit_at_k": source_hit,
        "term_coverage": round(term_coverage, 4),
        "found_terms": found_terms,
        "expected_sources": list(case.expected_sources),
        "returned_sources": sorted(paths),
        "result_count": len(results),
        "should_find": case.should_find,
    }


def run_benchmark(
    cases: Sequence[RetrievalCase],
    *,
    knowledge_dir: Path,
    output_dir: Path,
    top_k: int = 3,
) -> dict[str, Any]:
    """Run the actual builtin tool against a fixed local corpus."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    old_root = os.environ.get("POIROT_RAG_DIR")
    os.environ["POIROT_RAG_DIR"] = str(knowledge_dir)
    reset_knowledge_index()
    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            started = time.perf_counter()
            raw = knowledge_search.invoke({"query": case.query, "max_results": top_k})
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            try:
                payload = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                payload = {"results": [], "error": "tool returned invalid JSON"}
            item = evaluate_results(payload, case, top_k=top_k)
            item["query"] = case.query
            item["elapsed_ms"] = elapsed_ms
            if payload.get("error"):
                item["error"] = payload["error"]
            results.append(item)
    finally:
        reset_knowledge_index()
        if old_root is None:
            os.environ.pop("POIROT_RAG_DIR", None)
        else:
            os.environ["POIROT_RAG_DIR"] = old_root

    positive = [item for item in results if item["should_find"]]
    negative = [item for item in results if not item["should_find"]]
    source_hits = sum(item["source_hit_at_k"] for item in positive)
    negative_clean = sum(not item["result_count"] for item in negative)
    latencies = [float(item["elapsed_ms"]) for item in results]
    summary = {
        "schema_version": "1.0",
        "benchmark": "poirot-rag-retrieval-v1",
        "scope": "Curated local corpus; builtin knowledge_search; lexical BM25-style retrieval",
        "configuration": {
            "knowledge_dir": str(knowledge_dir),
            "case_count": len(cases),
            "top_k": top_k,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "aggregate": {
            "executed_cases": len(results),
            "passed_cases": sum(item["passed"] for item in results),
            "case_pass_rate": round(sum(item["passed"] for item in results) / len(results) * 100, 2) if results else 0.0,
            "recall_at_k": round(source_hits / len(positive) * 100, 2) if positive else 0.0,
            "term_coverage_rate": round(sum(item["term_coverage"] for item in positive) / len(positive) * 100, 2) if positive else 0.0,
            "negative_no_result_rate": round(negative_clean / len(negative) * 100, 2) if negative else 0.0,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        },
        "cases": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded local RAG retrieval benchmark.")
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    parser.add_argument("--knowledge-dir", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / ".poirot" / "evaluations" / "rag-retrieval-v1")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    summary = run_benchmark(
        read_cases(args.cases),
        knowledge_dir=args.knowledge_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    aggregate = summary["aggregate"]
    print(
        f"{aggregate['passed_cases']}/{aggregate['executed_cases']} passed | "
        f"Recall@{args.top_k}={aggregate['recall_at_k']}% | "
        f"term_coverage={aggregate['term_coverage_rate']}%"
    )
    print(f"summary: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
