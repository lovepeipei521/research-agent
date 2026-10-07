"""Regression coverage for the L2/L3 runtime wiring added after the base APIs."""
from __future__ import annotations

from unittest.mock import Mock

from poirot.backend.agents.multiagent.config import L2Config, L3Config, MultiAgentConfig
from poirot.backend.agents.multiagent.evolution.bootstrap import setup_l2
from poirot.backend.agents.multiagent.evolution.types import (
    ContextSummaryTemplate,
    FailureStats,
    SkillInjectionTemplate,
)
from poirot.backend.agents.multiagent.eval.types import DecisionLogRecord
from poirot.backend.agents.multiagent.metrics import MultiAgentMetricsStore
from poirot.backend.agents.multiagent.tools import make_specialist_tool
from poirot.backend.agents.multiagent.types import SpecialistRawResult, SpecialistResult


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


class _Model:
    def invoke(self, prompt: str) -> _Response:
        return _Response(
            '{"version":"v2","template_id":"default-context-summary",'
            '"extractors":[],"filters":[],"max_tokens":100,'
            '"prompt_skeleton":"{context}"}'
        )


def _config(tmp_path, *, l3: bool = False, sample_min: int = 1) -> MultiAgentConfig:
    return MultiAgentConfig(
        metrics_db_path=str(tmp_path / "multiagent.db"),
        l2=L2Config(enabled=True, eval_sample_min=sample_min, eval_sample_max=3),
        l3=L3Config(enabled=l3),
    )


def test_setup_l2_uses_runtime_model_seeds_templates_and_shares_l3_bridge(tmp_path):
    metrics = MultiAgentMetricsStore(str(tmp_path / "multiagent.db"))
    setup = setup_l2(_config(tmp_path, l3=True), metrics, model=_Model())

    assert setup is not None
    context_template = setup.version_dag.get_active(ContextSummaryTemplate)
    skill_template = setup.version_dag.get_active(SkillInjectionTemplate)
    assert context_template is not None
    assert skill_template is not None
    assert setup.l3_setup is not None
    assert setup.promotion_gate._bridge is setup.l3_setup.bridge

    outcome = setup.evolution_mutator.evolve_context_summary(
        context_template,
        FailureStats(by_category={}, dominant_category=None, sample_failures={}),
    )
    assert outcome.success is True


def test_worker_builds_eval_tasks_from_persisted_decision_logs(tmp_path):
    metrics = MultiAgentMetricsStore(str(tmp_path / "multiagent.db"))
    setup = setup_l2(_config(tmp_path, sample_min=2), metrics, model=_Model())
    assert setup is not None

    metrics.save_decision_log(
        DecisionLogRecord(
            log_id="failure", specialist_name="codex", task_id="failure-task",
            goal="repair test", success_criteria="test passes",
            success_criteria_met=0,
        )
    )
    metrics.save_decision_log(
        DecisionLogRecord(
            log_id="success", specialist_name="codex", task_id="success-task",
            goal="inspect module", success_criteria="summary returned",
            success_criteria_met=1,
        )
    )

    sample = setup.worker._load_task_sample("codex")
    assert {task.task_id for task in sample} == {"failure-task", "success-task"}


def test_active_templates_are_applied_to_specialist_request():
    context_template = ContextSummaryTemplate(
        version="v1", template_id="context", extractors=(), filters=(),
        max_tokens=100, prompt_skeleton="Delegated context: {context}",
    )
    skill_template = SkillInjectionTemplate(
        version="v1", template_id="skills", skill_selector=Mock(),
        injection_format="Follow the selected specialist procedure.", max_skills=3,
    )

    class _VersionDag:
        def get_active(self, artifact_type):
            if artifact_type is ContextSummaryTemplate:
                return context_template
            if artifact_type is SkillInjectionTemplate:
                return skill_template
            return None

    specialist = Mock()
    specialist.invoke.return_value = SpecialistRawResult(raw_output="raw")
    summarizer = Mock()
    summarizer.summarize.return_value = "original context"
    result_summarizer = Mock()
    result_summarizer.summarize.return_value = SpecialistResult(
        specialist_name="codex", summary="done", success=True,
    )

    tool = make_specialist_tool(
        "codex", specialist, summarizer, result_summarizer, version_dag=_VersionDag(),
    )
    tool.invoke({"goal": "g", "success_criteria": "sc"})

    request = specialist.invoke.call_args.args[0]
    assert request.context_summary == "Delegated context: original context"
    assert request.skill_injection == "Follow the selected specialist procedure."
