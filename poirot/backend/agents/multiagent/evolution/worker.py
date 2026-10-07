"""L2EvolutionWorker - daemon-thread cron worker + per-profile serial (Batch 11).

Design (42 doc S7.4 + spec.md L2EvolutionWorker Requirement):
- threading.Thread(daemon=True) + queue.Queue (reuse PiInstaller/DockerSandboxProvider pattern)
- worker loop: while not stop_event: task = queue.get(timeout); if task: run(task); else sleep(cron_interval)
- run(task): TriggerManager -> MetricsView -> FailureFocuser -> EvolutionMutator -> PromotionGate -> VersionDAG
- per-profile serial (daemon thread single-thread consume, INV-28)
- failure handling: catch + log + continue (no block) + 3 consecutive failures -> evolution_blocked
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from poirot.backend.agents.multiagent.evolution.evolution_mutator import EvolutionMutator
from poirot.backend.agents.multiagent.evolution.failure_focuser import FailureFocuser
from poirot.backend.agents.multiagent.evolution.metrics_l2 import OrchestrationMetricsL2
from poirot.backend.agents.multiagent.evolution.metrics_view import MetricsView
from poirot.backend.agents.multiagent.evolution.types import (
    EvolutionResult,
    EvolutionTask,
    FailureCategory,
    PromotionDecision,
)
from poirot.backend.agents.multiagent.evolution.version_dag import VersionDAG

logger = logging.getLogger(__name__)

_BLOCKED_AUTO_RELEASE_SECONDS = 86400  # 24h
_CONSECUTIVE_FAILURE_THRESHOLD = 3


class L2EvolutionWorker:
    """daemon-thread cron worker + per-profile serial (Batch 11).

    INVARIANT:
    - threading.Thread(daemon=True) + queue.Queue (no cron framework, INV-28)
    - per-profile serial (single daemon thread consume queue)
    - failure handling: catch + log + continue (no block, INV-15)
    - 3 consecutive failures -> evolution_blocked (INV-16)
    - 6h cron interval fallback (worker loop time.sleep)
    """

    def __init__(
        self,
        task_queue: "queue.Queue[EvolutionTask]",
        metrics_view: MetricsView,
        failure_focuser: FailureFocuser,
        evolution_mutator: EvolutionMutator,
        promotion_gate: Any,
        version_dag: VersionDAG,
        metrics_l2: OrchestrationMetricsL2,
        cron_interval_seconds: float = 21600.0,  # 6h
        eval_sample_min: int = 10,
        eval_sample_max: int = 15,
    ) -> None:
        self._queue = task_queue
        self._metrics_view = metrics_view
        self._focuser = failure_focuser
        self._mutator = evolution_mutator
        self._gate = promotion_gate
        self._version_dag = version_dag
        self._metrics_l2 = metrics_l2
        self._cron_interval = cron_interval_seconds
        self._eval_sample_min = eval_sample_min
        self._eval_sample_max = eval_sample_max

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure_counts: dict[str, int] = {}  # per failure_pattern

    def start(self) -> None:
        """Start daemon thread + worker loop."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop daemon thread (set stop_event)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def run(self, task: EvolutionTask) -> EvolutionResult:
        """Run single evolution task (per-profile serial by daemon thread).

        Orchestration: TriggerManager -> MetricsView -> FailureFocuser ->
        EvolutionMutator -> PromotionGate -> VersionDAG commit.
        """
        self._metrics_l2.record_evolution_start(
            experiment_id=task.task_id,
            artifact_type=task.artifact_type,
            from_version="",
        )

        try:
            result = self._run_evolution(task)
            return result
        except Exception as e:
            logger.exception("L2EvolutionWorker.run failed: %s", e)
            self._metrics_l2.record_evolution_failed(
                failure_type="evolution_error",
                experiment_id=task.task_id,
                detail=str(e),
            )
            self._record_consecutive_failure(task.profile, "evolution_error")
            return EvolutionResult(
                task_id=task.task_id,
                decision=PromotionDecision.FAILED,
                error=str(e),
            )

    def _run_evolution(self, task: EvolutionTask) -> EvolutionResult:
        """Orchestration closed loop: focus -> mutate -> evaluate -> gate -> commit."""
        # 1. focus
        failures = self._focuser.analyze(self._metrics_view, task.profile)

        if failures.dominant_category is None:
            # GOAL_UNCLEAR / SANDBOX_ISSUE dominant -> no evolution
            return EvolutionResult(
                task_id=task.task_id,
                decision=PromotionDecision.REJECT,
                rationale=f"non-evolvable dominant: {failures.dominant_category}",
            )

        # 2. get current active template
        if task.artifact_type == "skill_injection":
            from poirot.backend.agents.multiagent.evolution.types import SkillInjectionTemplate
            current = self._version_dag.get_active(SkillInjectionTemplate)
            if current is None:
                return EvolutionResult(
                    task_id=task.task_id,
                    decision=PromotionDecision.REJECT,
                    rationale="no active skill_injection template",
                )
            mutate_result = self._mutator.evolve_skill_injection(current, failures)
        else:
            from poirot.backend.agents.multiagent.evolution.types import ContextSummaryTemplate
            current = self._version_dag.get_active(ContextSummaryTemplate)
            if current is None:
                return EvolutionResult(
                    task_id=task.task_id,
                    decision=PromotionDecision.REJECT,
                    rationale="no active context_summary template",
                )
            mutate_result = self._mutator.evolve_context_summary(current, failures)

        # 3. mutator failed -> keep old is_active
        if not mutate_result.success:
            self._metrics_l2.record_evolution_failed(
                failure_type=mutate_result.failure_type,
                experiment_id=task.task_id,
                detail=mutate_result.rationale,
            )
            self._record_consecutive_failure(task.profile, mutate_result.failure_type)
            return EvolutionResult(
                task_id=task.task_id,
                decision=PromotionDecision.FAILED,
                error=mutate_result.failure_type,
                rationale=mutate_result.rationale,
            )

        # reset consecutive failure on success (clear all patterns for this profile)
        prefix = f"{task.profile}:"
        for key in list(self._failure_counts.keys()):
            if key.startswith(prefix):
                self._failure_counts.pop(key, None)

        # 4. Promotion needs actual historical tasks. There is no safe replay
        # executor for a candidate specialist template, so a missing sample is
        # a fail-closed rejection rather than a fabricated floor evaluation.
        task_sample = self._load_task_sample(task.profile)
        if len(task_sample) < self._eval_sample_min:
            return EvolutionResult(
                task_id=task.task_id,
                decision=PromotionDecision.REJECT,
                rationale=(
                    "insufficient historical decision-log samples "
                    f"({len(task_sample)}/{self._eval_sample_min})"
                ),
            )

        # 5. promotion gate (hash anti-loop + Wilson CI)
        candidate = mutate_result.candidate
        from poirot.backend.agents.multiagent.evolution.promotion_gate import EvalTask
        eval_result = self._gate.evaluate(candidate, current, task_sample)
        if not eval_result.success:
            self._metrics_l2.record_evolution_failed(
                failure_type=eval_result.failure_reason or "evaluation_failed",
                experiment_id=task.task_id,
                detail=eval_result.method_used,
            )
            return EvolutionResult(
                task_id=task.task_id,
                decision=PromotionDecision.FAILED,
                error=eval_result.failure_reason,
                rationale="candidate evaluation failed closed",
            )
        decision = self._gate.decide(candidate, current, eval_result)

        # 5. commit to VersionDAG
        if decision == PromotionDecision.ACCEPT:
            artifact_id = self._version_dag.commit(
                candidate,
                eval_result,
                trigger_source=task.trigger_source.value,
                trigger_detail=task.trigger_detail,
                rationale=mutate_result.rationale,
                decision="accept",
            )
            self._metrics_l2.record_promotion_decision(
                decision="accept",
                candidate_score=eval_result.candidate_score,
                baseline_score=eval_result.baseline_score,
                ci_low=eval_result.ci_low,
                ci_high=eval_result.ci_high,
            )
            return EvolutionResult(
                task_id=task.task_id,
                decision=PromotionDecision.ACCEPT,
                rationale=mutate_result.rationale,
            )
        else:
            # reject -> still commit to DAG (防重复尝试)
            self._version_dag.commit(
                candidate,
                eval_result,
                trigger_source=task.trigger_source.value,
                trigger_detail=task.trigger_detail,
                rationale=mutate_result.rationale,
                decision="reject",
            )
            self._metrics_l2.record_promotion_decision(
                decision="reject",
                candidate_score=eval_result.candidate_score,
                baseline_score=eval_result.baseline_score,
                ci_low=eval_result.ci_low,
                ci_high=eval_result.ci_high,
            )
            return EvolutionResult(
                task_id=task.task_id,
                decision=PromotionDecision.REJECT,
                rationale=mutate_result.rationale,
            )

    def _load_task_sample(self, specialist_name: str) -> list[Any]:
        """Build a bounded 80/20 failure/success sample from L3 decision logs."""
        reader = getattr(self._metrics_view, "get_decision_logs", None)
        if not callable(reader):
            return []
        try:
            records = reader(
                specialist_name,
                failure_category=None,
                limit=max(self._eval_sample_max * 5, self._eval_sample_min),
            )
        except Exception:
            logger.exception("Unable to load decision logs for %s", specialist_name)
            return []

        from poirot.backend.agents.multiagent.evolution.promotion_gate import EvalTask

        failures = [record for record in records if record.success_criteria_met != 1]
        successes = [record for record in records if record.success_criteria_met == 1]
        failure_quota = min(len(failures), max(1, round(self._eval_sample_max * 0.8)))
        success_quota = min(len(successes), self._eval_sample_max - failure_quota)
        selected = failures[:failure_quota] + successes[:success_quota]
        if len(selected) < self._eval_sample_min:
            return []
        return [
            EvalTask(
                task_id=record.task_id,
                goal=record.goal,
                success_criteria=record.success_criteria,
                expected_outcome=record.lesson_text,
            )
            for record in selected
        ]

    def _worker_loop(self) -> None:
        """Daemon thread worker loop: consume queue + 6h cron fallback."""
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=1.0)
            except queue.Empty:
                # 6h cron fallback (sleep short to allow stop_event check)
                self._stop_event.wait(timeout=min(self._cron_interval, 60.0))
                continue

            try:
                self.run(task)
            except Exception as e:
                logger.exception("L2EvolutionWorker worker_loop task failed: %s", e)

    def _record_consecutive_failure(self, profile: str, failure_type: str) -> None:
        """Track consecutive failures, mark blocked after 3 (INV-16)."""
        key = f"{profile}:{failure_type}"
        self._failure_counts[key] = self._failure_counts.get(key, 0) + 1
        if self._failure_counts[key] >= _CONSECUTIVE_FAILURE_THRESHOLD:
            auto_release = time.time() + _BLOCKED_AUTO_RELEASE_SECONDS
            self._metrics_l2.record_blocked_marked(
                blocked_pattern=key,
                blocked_type="evolution",
                auto_release_at=str(int(auto_release)),
            )
            # reset after marking (24h auto release)
            self._failure_counts[key] = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
