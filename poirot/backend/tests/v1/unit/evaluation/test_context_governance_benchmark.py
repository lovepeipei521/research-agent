from pathlib import Path

from poirot.backend.evaluation.context_governance_benchmark import read_cases, run_benchmark


def test_context_governance_benchmark_runs_controlled_cases(tmp_path: Path) -> None:
    cases = read_cases(Path("resource/poirot_context_governance_eval_v1.json"))

    summary = run_benchmark(cases, output_dir=tmp_path)

    assert summary["aggregate"]["executed_cases"] == 10
    failed = [item for item in summary["cases"] if not item["passed"]]
    assert failed == []
    assert summary["aggregate"]["externalization_integrity_rate"] == 100.0
    assert summary["aggregate"]["p5_tool_stop_pass_rate"] == 100.0
    assert (tmp_path / "summary.json").is_file()
