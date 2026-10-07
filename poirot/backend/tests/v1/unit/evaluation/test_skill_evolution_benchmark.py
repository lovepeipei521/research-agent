from pathlib import Path

from poirot.backend.evaluation.skill_evolution_benchmark import read_cases, run_benchmark


def test_skill_evolution_benchmark_runs_controlled_cases(tmp_path: Path) -> None:
    cases = read_cases(Path("resource/poirot_skill_evolution_eval_v1.json"))

    summary = run_benchmark(cases, output_dir=tmp_path)

    assert summary["aggregate"]["executed_cases"] == 12
    assert [item for item in summary["cases"] if not item["passed"]] == []
    assert summary["aggregate"]["trigger_contract_pass_rate"] == 100.0
    assert summary["aggregate"]["gate_contract_pass_rate"] == 100.0
    assert summary["aggregate"]["rollback_contract_pass_rate"] == 100.0
    assert summary["aggregate"]["sqlite_metrics_persistence_pass_rate"] == 100.0
    assert (tmp_path / "summary.json").is_file()
