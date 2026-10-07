import json
from pathlib import Path

from poirot.backend.evaluation.rag_benchmark import (
    RetrievalCase,
    evaluate_results,
    read_cases,
    run_benchmark,
)


def test_read_rag_cases() -> None:
    cases = read_cases(Path("resource/poirot_rag_eval_v1.json"))

    assert len(cases) == 20
    assert cases[0].expected_sources == ("sandbox.md",)
    assert sum(case.should_find is False for case in cases) == 2


def test_evaluate_results_checks_source_and_terms() -> None:
    case = RetrievalCase(
        case_id="c1",
        query="sandbox",
        expected_sources=("sandbox.md",),
        required_terms=("Docker Sandbox",),
        should_find=True,
    )
    result = evaluate_results(
        {"results": [{"path": "sandbox.md", "title": "Sandbox", "snippet": "Docker Sandbox"}]},
        case,
        top_k=3,
    )

    assert result["passed"] is True
    assert result["source_hit_at_k"] is True
    assert result["term_coverage"] == 1.0


def test_evaluate_results_matches_terms_split_by_line_wrap() -> None:
    case = RetrievalCase(
        case_id="wrapped",
        query="Chroma",
        expected_sources=("facts.md",),
        required_terms=("does not use Chroma",),
        should_find=True,
    )

    result = evaluate_results(
        {"results": [{"path": "facts.md", "snippet": "This version does\nnot use Chroma."}]},
        case,
        top_k=3,
    )

    assert result["passed"] is True


def test_retrieval_runner_writes_metrics_and_restores_env(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "facts.md").write_text("# Facts\nBM25 local retrieval", encoding="utf-8")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps({
            "cases": [
                {"id": "hit", "query": "BM25", "expected_sources": ["facts.md"], "required_terms": ["BM25"], "should_find": True},
                {"id": "miss", "query": "unavailable fact", "expected_sources": [], "required_terms": [], "should_find": False},
            ]
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("POIROT_RAG_DIR", "original-root")

    summary = run_benchmark(
        read_cases(cases_path),
        knowledge_dir=corpus,
        output_dir=tmp_path / "out",
    )

    assert summary["aggregate"]["recall_at_k"] == 100.0
    assert summary["aggregate"]["negative_no_result_rate"] == 100.0
    assert (tmp_path / "out" / "summary.json").exists()
    assert __import__("os").environ["POIROT_RAG_DIR"] == "original-root"
