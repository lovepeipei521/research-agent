"""Controlled mechanism evaluation for the Skill evolution control plane."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from poirot.backend.agents.skill.evolution.gates.git_ratchet import GitRatchet
from poirot.backend.agents.skill.evolution.gates.score_delta_gate import ScoreDeltaGate
from poirot.backend.agents.skill.evolution.triggers.metric_monitor import (
    MetricMonitorTrigger,
)
from poirot.backend.agents.skill.evolution.types import EvalEvidence, EvalResult
from poirot.backend.agents.skill.store import SQLiteSkillStore
from poirot.backend.agents.skill.types import SkillLineage, SkillRecord

_PROJECT_ROOT = Path(__file__).parents[3]
_DEFAULT_CASES = _PROJECT_ROOT / "resource" / "poirot_skill_evolution_eval_v1.json"


@dataclass(frozen=True)
class EvolutionCase:
    case_id: str
    kind: str
    data: dict[str, Any]


def read_cases(path: Path) -> list[EvolutionCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = raw.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("skill evolution case file must contain a non-empty cases list")
    parsed: list[EvolutionCase] = []
    for item in cases:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("kind"), str):
            raise ValueError("each skill evolution case requires id and kind")
        parsed.append(EvolutionCase(item["id"], item["kind"], item))
    return parsed


def _skill_record(
    case_id: str,
    *,
    name: str,
    role: str,
    parent_skill_id: str | None = None,
) -> SkillRecord:
    is_candidate = role == "candidate"
    lineage = SkillLineage(
        parent_skill_ids=(parent_skill_id,) if parent_skill_id else (),
        generation=1 if is_candidate else 0,
        origin="FIXED" if is_candidate else "IMPORTED",
        version_hash=f"hash-{case_id}-{role}",
        created_by="controlled-benchmark",
    )
    return SkillRecord(
        skill_id=f"{case_id}__{role}",
        name=name,
        path=f"controlled://{case_id}/{role}/SKILL.md",
        content_hash=f"hash-{case_id}-{role}",
        lineage=lineage,
        description="Controlled evaluation fixture.",
    )


def _expected_counters(events: dict[str, Any]) -> dict[str, int]:
    selections = int(events["selections"])
    outcomes = events.get("outcomes", [])
    if not isinstance(outcomes, list):
        raise ValueError("events.outcomes must be a list")
    count = 0
    applied = 0
    completions = 0
    fallbacks = 0
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise ValueError("each outcome must be an object")
        repetitions = int(outcome.get("count", 1))
        if repetitions < 0:
            raise ValueError("outcome count cannot be negative")
        applied_value = outcome.get("applied")
        if applied_value not in (True, False, None):
            raise ValueError("outcome applied must be true, false, or null")
        completed = bool(outcome.get("task_completed", False))
        count += repetitions
        if applied_value is True:
            applied += repetitions
            if completed:
                completions += repetitions
        elif applied_value is False and not completed:
            fallbacks += repetitions
    if count != selections:
        raise ValueError(f"outcome count ({count}) must equal selections ({selections})")
    return {
        "selections": selections,
        "applied": applied,
        "completions": completions,
        "fallbacks": fallbacks,
    }


def _seed_events(
    store: SQLiteSkillStore,
    skill_id: str,
    events: dict[str, Any],
    *,
    run_prefix: str,
) -> tuple[SkillRecord, bool]:
    expected = _expected_counters(events)
    for _ in range(expected["selections"]):
        store.record_selection(skill_id)
    sequence = 0
    for outcome in events["outcomes"]:
        for _ in range(int(outcome.get("count", 1))):
            store.record_outcome(
                skill_id,
                f"{run_prefix}-{sequence}",
                outcome.get("applied"),
                bool(outcome.get("task_completed", False)),
                note="controlled mechanism evaluation",
            )
            sequence += 1
    stored = store.get(skill_id)
    if stored is None:
        raise RuntimeError(f"skill disappeared from store: {skill_id}")
    persisted = _matches_counters(stored, expected)
    return stored, persisted


def _append_events(
    store: SQLiteSkillStore,
    skill_id: str,
    events: dict[str, Any],
    *,
    run_prefix: str,
) -> SkillRecord:
    expected = _expected_counters(events)
    for _ in range(expected["selections"]):
        store.record_selection(skill_id)
    sequence = 0
    for outcome in events["outcomes"]:
        for _ in range(int(outcome.get("count", 1))):
            store.record_outcome(
                skill_id,
                f"{run_prefix}-{sequence}",
                outcome.get("applied"),
                bool(outcome.get("task_completed", False)),
                note="controlled mechanism evaluation",
            )
            sequence += 1
    stored = store.get(skill_id)
    if stored is None:
        raise RuntimeError(f"skill disappeared from store: {skill_id}")
    return stored


def _matches_counters(record: SkillRecord, expected: dict[str, int]) -> bool:
    return (
        record.total_selections == expected["selections"]
        and record.total_applied == expected["applied"]
        and record.total_completions == expected["completions"]
        and record.total_fallbacks == expected["fallbacks"]
    )


def _with_store(case: EvolutionCase, output_dir: Path) -> SQLiteSkillStore:
    case_dir = output_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    return SQLiteSkillStore(case_dir / "skills.db")


def _run_trigger_case(case: EvolutionCase, output_dir: Path) -> dict[str, Any]:
    store = _with_store(case, output_dir)
    try:
        baseline = _skill_record(case.case_id, name=str(case.data["skill_name"]), role="baseline")
        store.register(baseline)
        stored, persistence_ok = _seed_events(store, baseline.skill_id, case.data["events"], run_prefix=case.case_id)
        config = case.data["trigger"]
        trigger = MetricMonitorTrigger(
            min_selections=int(config["min_selections"]),
            cooldown_turns=int(config["cooldown_turns"]),
            llm=None,
        )
        contexts = trigger.should_trigger(store)
        actual_triggered = bool(contexts)
        direction = contexts[0].fix_direction if contexts else ""
        expected = case.data["expect"]
        direction_ok = "direction_contains" not in expected or str(expected["direction_contains"]) in direction
        passed = actual_triggered is bool(expected["triggered"]) and direction_ok and persistence_ok
        return {
            "id": case.case_id,
            "kind": case.kind,
            "passed": passed,
            "triggered": actual_triggered,
            "direction": direction,
            "metrics_persistence_pass": persistence_ok,
            "metrics": _metrics_payload(stored),
        }
    finally:
        store.close()


def _run_cooldown_case(case: EvolutionCase, output_dir: Path) -> dict[str, Any]:
    store = _with_store(case, output_dir)
    try:
        baseline = _skill_record(case.case_id, name=str(case.data["skill_name"]), role="baseline")
        store.register(baseline)
        initial, initial_persistence_ok = _seed_events(store, baseline.skill_id, case.data["events"], run_prefix=f"{case.case_id}-initial")
        config = case.data["trigger"]
        trigger = MetricMonitorTrigger(
            min_selections=int(config["min_selections"]),
            cooldown_turns=int(config["cooldown_turns"]),
            llm=None,
        )
        initial_triggered = bool(trigger.should_trigger(store))
        trigger.mark_evolved(initial.name, initial.total_selections)
        blocked_during_cooldown = not trigger.should_trigger(store)
        updated = _append_events(
            store,
            baseline.skill_id,
            case.data["additional_events"],
            run_prefix=f"{case.case_id}-additional",
        )
        additional_persistence_ok = _matches_counters(
            updated,
            _expected_counters(_combined_events(case.data["events"], case.data["additional_events"])),
        )
        retriggered_after_cooldown = bool(trigger.should_trigger(store))
        expected = case.data["expect"]
        passed = (
            initial_triggered is bool(expected["initial_triggered"])
            and blocked_during_cooldown is bool(expected["blocked_during_cooldown"])
            and retriggered_after_cooldown is bool(expected["retriggered_after_cooldown"])
            and initial_persistence_ok
            and additional_persistence_ok
        )
        return {
            "id": case.case_id,
            "kind": case.kind,
            "passed": passed,
            "initial_triggered": initial_triggered,
            "blocked_during_cooldown": blocked_during_cooldown,
            "retriggered_after_cooldown": retriggered_after_cooldown,
            "metrics_persistence_pass": initial_persistence_ok and additional_persistence_ok,
            "metrics": _metrics_payload(updated),
        }
    finally:
        store.close()


def _combined_events(initial: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "selections": int(initial["selections"]) + int(extra["selections"]),
        "outcomes": [*initial["outcomes"], *extra["outcomes"]],
    }


def _run_gate_case(case: EvolutionCase) -> dict[str, Any]:
    candidate = _skill_record(case.case_id, name="controlled-gate", role="candidate")
    has_baseline = bool(case.data["has_baseline"])
    baseline = _skill_record(case.case_id, name="controlled-gate", role="baseline") if has_baseline else None
    evidence = tuple(
        EvalEvidence(
            kind="programmatic_rule",
            rule_name=f"rule-{index + 1}",
            baseline_pass=bool(passed),
            candidate_pass=True,
            detail="controlled score-delta fixture",
        )
        for index, passed in enumerate(case.data["baseline_passes"])
    )
    result = EvalResult(
        score=float(case.data["candidate_score"]),
        hard_failures=tuple(str(item) for item in case.data["hard_failures"]),
        evidence=evidence,
    )
    decision = ScoreDeltaGate().decide(candidate, baseline, result)
    expected = case.data["expect"]
    reason_ok = "reason_contains" not in expected or str(expected["reason_contains"]) in decision.reason
    baseline_score = (
        round(sum(item.baseline_pass for item in evidence) / len(evidence), 2)
        if evidence
        else None
    )
    return {
        "id": case.case_id,
        "kind": case.kind,
        "passed": decision.recommendation == expected["recommendation"] and reason_ok,
        "recommendation": decision.recommendation,
        "reason": decision.reason,
        "candidate_score": result.score,
        "baseline_score": baseline_score,
        "hard_failures": list(result.hard_failures),
    }


def _run_ratchet_case(case: EvolutionCase, output_dir: Path) -> dict[str, Any]:
    store = _with_store(case, output_dir)
    try:
        name = str(case.data["skill_name"])
        baseline = _skill_record(case.case_id, name=name, role="baseline")
        candidate = _skill_record(
            case.case_id,
            name=name,
            role="candidate",
            parent_skill_id=baseline.skill_id,
        )
        store.register(baseline)
        store.create_version(baseline.skill_id, candidate, "FIXED")
        current, persistence_ok = _seed_events(store, candidate.skill_id, case.data["events"], run_prefix=case.case_id)
        config = case.data["ratchet"]
        rollback_target = GitRatchet(
            degradation_threshold=float(config["degradation_threshold"]),
            min_selections=int(config["min_selections"]),
        ).check_and_rollback(store, current)
        active = store.get_active(name)
        versions = store.get_versions(name)
        expected = case.data["expect"]
        expected_id = baseline.skill_id if expected["active_version"] == "baseline" else candidate.skill_id
        rolled_back = rollback_target is not None
        versions_preserved = len(versions) == 2
        passed = (
            rolled_back is bool(expected["rolled_back"])
            and active is not None
            and active.skill_id == expected_id
            and versions_preserved
            and persistence_ok
        )
        return {
            "id": case.case_id,
            "kind": case.kind,
            "passed": passed,
            "rolled_back": rolled_back,
            "rollback_target": rollback_target,
            "active_skill_id": active.skill_id if active else None,
            "versions_preserved": versions_preserved,
            "metrics_persistence_pass": persistence_ok,
            "metrics": _metrics_payload(current),
        }
    finally:
        store.close()


def _metrics_payload(record: SkillRecord) -> dict[str, Any]:
    return {
        "selections": record.total_selections,
        "applied": record.total_applied,
        "completions": record.total_completions,
        "fallbacks": record.total_fallbacks,
        "effective_rate": round(record.effective_rate, 4),
        "fallback_rate": round(record.fallback_rate, 4),
    }


def run_benchmark(cases: Sequence[EvolutionCase], *, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for case in cases:
        if case.kind == "trigger":
            results.append(_run_trigger_case(case, output_dir))
        elif case.kind == "cooldown":
            results.append(_run_cooldown_case(case, output_dir))
        elif case.kind == "gate":
            results.append(_run_gate_case(case))
        elif case.kind == "ratchet":
            results.append(_run_ratchet_case(case, output_dir))
        else:
            raise ValueError(f"unsupported skill evolution case kind: {case.kind}")

    trigger_cases = [item for item in results if item["kind"] in {"trigger", "cooldown"}]
    gate_cases = [item for item in results if item["kind"] == "gate"]
    ratchet_cases = [item for item in results if item["kind"] == "ratchet"]
    persistence_cases = [item for item in results if "metrics_persistence_pass" in item]
    summary = {
        "schema_version": "1.0",
        "benchmark": "poirot-skill-evolution-v1",
        "scope": "Controlled mechanism evaluation of SQLite metric persistence, MetricMonitor triggering, ScoreDeltaGate, and GitRatchet. It does not measure post-mutation business-task quality.",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "aggregate": {
            "executed_cases": len(results),
            "passed_cases": sum(item["passed"] for item in results),
            "mechanism_pass_rate": round(sum(item["passed"] for item in results) / len(results) * 100, 2) if results else 0.0,
            "trigger_contract_pass_rate": _pass_rate(trigger_cases),
            "gate_contract_pass_rate": _pass_rate(gate_cases),
            "rollback_contract_pass_rate": _pass_rate(ratchet_cases),
            "sqlite_metrics_persistence_pass_rate": round(
                sum(item["metrics_persistence_pass"] for item in persistence_cases) / len(persistence_cases) * 100,
                2,
            ) if persistence_cases else None,
        },
        "cases": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _pass_rate(items: Sequence[dict[str, Any]]) -> float | None:
    return round(sum(item["passed"] for item in items) / len(items) * 100, 2) if items else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run controlled Skill evolution mechanism evaluation.")
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / ".poirot" / "evaluations" / "skill-evolution-v1",
    )
    args = parser.parse_args(argv)
    summary = run_benchmark(read_cases(args.cases), output_dir=args.output_dir)
    aggregate = summary["aggregate"]
    print(
        f"{aggregate['passed_cases']}/{aggregate['executed_cases']} passed | "
        f"trigger={aggregate['trigger_contract_pass_rate']}% | "
        f"gate={aggregate['gate_contract_pass_rate']}% | "
        f"rollback={aggregate['rollback_contract_pass_rate']}%"
    )
    print(f"summary: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
