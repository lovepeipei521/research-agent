"""tools.py — 动态生成 specialist tool + subagent tool。

设计（spec.md tools.py Requirement + design.md §3）:
- make_specialist_tool(name, specialist, context_summarizer, result_summarizer) → BaseTool
- make_subagent_tool(subagent_provider, context_summarizer, result_summarizer) → BaseTool
- tool schema 精简：LLM 只填 3 参数（goal + success_criteria + sandbox_id 可选）
- tool handler 内部编排：ContextSummarizer → specialist.invoke → ResultSummarizer → 返 JSON
- specialist 失败返 error JSON（LLM 决策 retry/fallback，INV#6 pairing 完整性）
- 新增 specialist 只注册到 SpecialistRegistry，factory 自动生成 tool，零代码侵入
- ThreadState via ContextVar（OrchestrationMiddleware Batch 9 设置）
"""
from __future__ import annotations

import contextvars
import json
import time
from typing import Any

from langchain_core.tools import BaseTool, tool

from poirot.backend.agents.multiagent.context_summarizer import ContextSummarizer
from poirot.backend.agents.multiagent.exceptions import (
    SpecialistError,
    SubagentError,
)
from poirot.backend.agents.multiagent.result_summarizer import ResultSummarizer
from poirot.backend.agents.multiagent.specialist import SpecialistAgent
from poirot.backend.agents.multiagent.subagent import SubagentProvider
from poirot.backend.agents.multiagent.types import (
    SpecialistRequest,
    SubagentRequest,
)

_current_state: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "multiagent_state", default=None,
)
_current_thread_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "multiagent_thread_id", default=None,
)


def set_current_state(state: dict) -> None:
    """OrchestrationMiddleware 调用：设置当前 ThreadState 供 tool handler 读取。"""
    _current_state.set(state)


def set_current_runtime_config(config: Any | None) -> None:
    """Store the current LangGraph tool config for specialist lazy resources.

    The graph state intentionally does not contain runtime-only values such as
    ``configurable.thread_id``.  Specialist tools nevertheless need that ID to
    acquire the shared sandbox, so the orchestration middleware passes a small
    per-call context here instead of persisting it in ThreadState.
    """
    configurable = config.get("configurable") if isinstance(config, dict) else None
    thread_id = configurable.get("thread_id") if isinstance(configurable, dict) else None
    _current_thread_id.set(str(thread_id) if thread_id else None)


def clear_current_runtime_context() -> None:
    """Clear request-scoped runtime data after a tool call."""
    _current_thread_id.set(None)


def get_current_state() -> dict:
    """tool handler 调用：获取当前 ThreadState（未设置时返空 dict）。"""
    return _current_state.get() or {}


def _runtime_thread_id(state: dict) -> str | None:
    return _current_thread_id.get()


def _extract_sandbox_id(state: dict, sandbox_id: str | None) -> str | None:
    if sandbox_id is not None:
        return sandbox_id
    sandbox = state.get("sandbox")
    if isinstance(sandbox, dict):
        return sandbox.get("sandbox_id")
    return None


def _ensure_specialist_sandbox(state: dict, requested_id: str | None) -> str | None:
    """Resolve or lazily acquire the shared thread sandbox for specialists."""
    resolved = _extract_sandbox_id(state, requested_id)
    if resolved:
        return resolved
    try:
        from poirot.backend.agents.sandbox.integration.context import get_sandbox_id
        current = get_sandbox_id()
        if current:
            return current
    except Exception:
        pass
    provider = state.get("_sandbox_provider")
    thread_id = (
        state.get("metadata", {}).get("thread_id")
        or state.get("thread_id")
        or _runtime_thread_id(state)
    )
    if provider is not None and thread_id:
        try:
            return provider.acquire(thread_id, user_id=state.get("metadata", {}).get("user_id", "default-user"))
        except Exception:
            return None
    return None


def _write_decision_log_async(
    writer: Any, specialist_name: str, goal: str, success_criteria: str, result: Any,
) -> None:
    """异步写 DecisionLogRecord（L3 扩展，fire-and-forget）.

    lazy import L3 类型避免循环依赖（同 L1 metrics.py pattern）.
    failure_category 从 result 获取（L2 ResultSummarizer 输出，可能为 None）.
    """
    import uuid
    from poirot.backend.agents.journal.events import utc_now_iso
    from poirot.backend.agents.multiagent.eval.types import DecisionLogRecord
    from poirot.backend.agents.multiagent.evolution.types import FailureCategory

    failure_category = getattr(result, "failure_category", None)
    if isinstance(failure_category, str):
        try:
            failure_category = FailureCategory(failure_category)
        except ValueError:
            failure_category = None

    record = DecisionLogRecord(
        log_id=str(uuid.uuid4()),
        specialist_name=specialist_name,
        task_id=str(uuid.uuid4()),
        goal=goal,
        success_criteria=success_criteria,
        failure_category=failure_category,
        success_criteria_met=1 if _resolve_success(result) else 0,
        lesson_text=None,  # MVP 不生成 lesson，L3 后置分析留 follow-up
        timestamp=utc_now_iso(),
    )
    writer.write_async(record)


def _apply_context_template(
    context_summary: str,
    template: Any | None,
    goal: str,
    success_criteria: str,
) -> str:
    """Apply the persisted, hot-swappable fields shared by all summarizers."""
    if template is None:
        return context_summary
    skeleton = getattr(template, "prompt_skeleton", "")
    if skeleton:
        rendered = (
            str(skeleton)
            .replace("{goal}", goal)
            .replace("{success_criteria}", success_criteria)
            .replace("{context}", context_summary)
        )
        context_summary = (
            rendered if "{context}" in str(skeleton) else f"{rendered}\n\n{context_summary}"
        )
    max_tokens = getattr(template, "max_tokens", None)
    if isinstance(max_tokens, int) and max_tokens > 0:
        # A conservative 4 chars/token bound keeps provider-neutral requests bounded.
        max_chars = max_tokens * 4
        if len(context_summary) > max_chars:
            context_summary = context_summary[:max_chars] + "\n...(truncated)"
    return context_summary


def _resolve_success(result: Any) -> bool:
    """Resolve the canonical outcome while keeping legacy result objects valid."""
    raw_execution_status = getattr(result, "execution_status", None)
    raw_criteria_status = getattr(result, "criteria_status", None)
    execution_status = raw_execution_status if isinstance(raw_execution_status, str) else None
    criteria_status = raw_criteria_status if isinstance(raw_criteria_status, str) else None
    legacy_success = bool(getattr(result, "success", False))
    if execution_status == "completed" and criteria_status == "met":
        return True
    if execution_status in {"failed", "cancelled", "timeout"} or criteria_status == "not_met":
        return False
    return legacy_success


def _result_payload(result: Any, *, specialist_name: str | None = None) -> dict[str, Any]:
    usage = getattr(result, "usage", None)
    failure_category = getattr(result, "failure_category", None)
    if not isinstance(failure_category, str):
        value = getattr(failure_category, "value", None)
        failure_category = value if isinstance(value, str) else None

    def _text(value: Any) -> str:
        return value if isinstance(value, str) else ""

    def _number(value: Any) -> float:
        return float(value) if isinstance(value, (int, float)) else 0.0

    def _count(value: Any) -> int:
        return int(value) if isinstance(value, int) else 0

    artifacts = getattr(result, "artifacts", ())
    artifact_items = artifacts if isinstance(artifacts, (tuple, list)) else ()
    raw_execution_status = getattr(result, "execution_status", None)
    raw_criteria_status = getattr(result, "criteria_status", None)
    execution_status = raw_execution_status if isinstance(raw_execution_status, str) else None
    criteria_status = raw_criteria_status if isinstance(raw_criteria_status, str) else None
    resolved_success = _resolve_success(result)
    return {
        "success": resolved_success,
        "execution_status": execution_status,
        "criteria_status": criteria_status,
        "summary": _text(getattr(result, "summary", "")),
        "specialist": specialist_name or _text(getattr(result, "specialist_name", "")),
        "gap_analysis": _text(getattr(result, "gap_analysis", "")),
        "failure_category": failure_category,
        "duration_seconds": _number(getattr(result, "duration_seconds", 0.0)),
        "usage": {
            "prompt_tokens": _count(getattr(usage, "prompt_tokens", 0)),
            "completion_tokens": _count(getattr(usage, "completion_tokens", 0)),
            "total_tokens": _count(getattr(usage, "total_tokens", 0)),
        },
        "artifacts": [
            {"path": artifact.path, "type": artifact.artifact_type}
            for artifact in artifact_items
        ],
    }


def make_specialist_tool(
    name: str,
    specialist: SpecialistAgent,
    context_summarizer: ContextSummarizer,
    result_summarizer: ResultSummarizer,
    *,
    max_steps: int = 50,
    timeout_seconds: int = 600,
    version_dag: Any | None = None,
    budget_guard: Any | None = None,
    decision_log_writer: Any | None = None,
    sandbox_provider: Any | None = None,
) -> BaseTool:
    """Factory：为 specialist 动态生成 delegate_to_<name> tool。

    tool handler 内部编排：ContextSummarizer → specialist.invoke → ResultSummarizer → JSON。
    LLM 只填 3 参数（goal + success_criteria + sandbox_id 可选），其余内部处理。
    L2 扩展：version_dag 非 None 时读 is_active SkillInjectionTemplate；
    budget_guard 非 None 时 check_and_record，超限返 BudgetExceeded JSON。
    L3 扩展：decision_log_writer 非 None 时异步写 DecisionLogRecord（fire-and-forget）。
    """

    @tool(f"delegate_to_{name}")
    def delegate_tool(
        goal: str,
        success_criteria: str,
        sandbox_id: str | None = None,
    ) -> str:
        """Delegate task to specialist. Provide goal and success_criteria. sandbox_id optional (uses thread sandbox if omitted)."""
        state = get_current_state()
        resolved_sandbox_id = _extract_sandbox_id(state, sandbox_id)
        if resolved_sandbox_id is None:
            try:
                from poirot.backend.agents.sandbox.integration.context import get_sandbox_id
                resolved_sandbox_id = get_sandbox_id()
            except Exception:
                resolved_sandbox_id = None
        thread_id = (
            state.get("metadata", {}).get("thread_id")
            or state.get("thread_id")
            or _runtime_thread_id(state)
        )
        if resolved_sandbox_id is None and sandbox_provider is not None and not thread_id:
            return json.dumps({
                "success": False,
                "execution_status": "failed",
                "criteria_status": "unknown",
                "error": {
                    "type": "SandboxStartupError",
                    "message": "sandbox provider is configured but runtime thread_id is missing",
                },
                "gap_analysis": "Cannot acquire the thread sandbox without a runtime thread_id.",
                "suggestion": "Do not retry blindly; inspect the graph runtime configuration.",
            })
        if resolved_sandbox_id is None and sandbox_provider is not None and thread_id:
            try:
                resolved_sandbox_id = sandbox_provider.acquire(
                    thread_id,
                    user_id=state.get("metadata", {}).get("user_id", "default-user"),
                )
            except Exception as exc:
                return json.dumps({
                    "success": False,
                    "execution_status": "failed",
                    "criteria_status": "unknown",
                    "error": {"type": "SandboxStartupError", "message": str(exc)},
                    "gap_analysis": "Could not acquire the thread sandbox for the specialist.",
                    "suggestion": "Check sandbox provider configuration and retry.",
                })

        # L2 BudgetGuard check (budget_guard 非 None 时，超限返 BudgetExceeded JSON)
        if budget_guard is not None:
            from types import SimpleNamespace
            cost = SimpleNamespace(tokens=0, cost_usd=0.0, calls=1)
            budget_result = budget_guard.check_and_record(name, cost)
            if not budget_result.allowed:
                return json.dumps({
                    "success": False,
                    "execution_status": "failed",
                    "criteria_status": "unknown",
                    "error": {
                        "type": "BudgetExceeded",
                        "message": f"{name} budget exceeded: {budget_result.reason}",
                        "remaining": {
                            "tokens": budget_result.remaining.tokens if budget_result.remaining else 0,
                            "cost_usd": budget_result.remaining.cost_usd if budget_result.remaining else 0.0,
                            "calls": budget_result.remaining.calls if budget_result.remaining else 0,
                        },
                        "fallback_target": "lead",
                    },
                    "suggestion": f"{name} daily budget exceeded, lead agent should execute task directly or wait UTC 0 reset.",
                })

        context_template = None
        skill_template = None
        if version_dag is not None:
            from poirot.backend.agents.multiagent.evolution.types import (
                ContextSummaryTemplate,
                SkillInjectionTemplate,
            )

            context_template = version_dag.get_active(ContextSummaryTemplate)
            skill_template = version_dag.get_active(SkillInjectionTemplate)
        context_summary = context_summarizer.summarize(
            state, goal, success_criteria, template=context_template
        )
        context_summary = _apply_context_template(
            context_summary, context_template, goal, success_criteria
        )

        request = SpecialistRequest(
            goal=goal,
            success_criteria=success_criteria,
            context_summary=context_summary,
            sandbox_id=resolved_sandbox_id,
            artifacts_path=state.get("metadata", {}).get("artifacts_path"),
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            skill_injection=(
                getattr(skill_template, "injection_format", None)
                if skill_template is not None
                else None
            ),
        )

        started_at = time.monotonic()
        try:
            raw = specialist.invoke(request)
        except SpecialistError as e:
            payload = {
                "success": False,
                "execution_status": "failed",
                "criteria_status": "unknown",
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
                "gap_analysis": str(e),
                "duration_seconds": time.monotonic() - started_at,
                "suggestion": "retry, fallback to another specialist, or self-do",
            }
            if decision_log_writer is not None:
                _write_decision_log_async(
                    decision_log_writer,
                    name,
                    goal,
                    success_criteria,
                    type("FailedResult", (), {"success": False, "failure_category": None})(),
                )
            return json.dumps(payload)

        # A runtime can return successful tool evidence without a prose
        # response (for example, an empty-directory listing). Let the result
        # summarizer evaluate that case; only reject a truly empty response
        # with no evidence at all.
        if not raw.raw_output.strip() and not raw.evidence:
            return json.dumps({
                "success": False,
                "execution_status": "failed",
                "criteria_status": "unknown",
                "error": {
                    "type": "SpecialistEmptyOutput",
                    "message": "specialist returned no assistant output",
                },
                "gap_analysis": "specialist returned no assistant output",
                "duration_seconds": raw.duration_seconds,
                "suggestion": "inspect specialist startup/RPC logs; do not retry blindly",
            })

        try:
            result = result_summarizer.summarize(
                raw.raw_output,
                list(raw.artifacts),
                goal,
                success_criteria,
                evidence=list(raw.evidence),
            )
        except TypeError as exc:
            # Preserve compatibility with user-provided four-argument
            # summarizers while built-in summarizers consume runtime evidence.
            if "evidence" not in str(exc):
                raise
            result = result_summarizer.summarize(
                raw.raw_output, list(raw.artifacts), goal, success_criteria
            )

        # L3 扩展：异步写 DecisionLogRecord（fire-and-forget，不阻塞 L1 turn）
        if decision_log_writer is not None:
            _write_decision_log_async(decision_log_writer, name, goal, success_criteria, result)

        return json.dumps(_result_payload(result))

    return delegate_tool


def make_subagent_tool(
    subagent_provider: SubagentProvider,
    context_summarizer: ContextSummarizer,
    result_summarizer: ResultSummarizer,
    *,
    max_steps: int = 20,
    timeout_seconds: int = 300,
    decision_log_writer: Any | None = None,
) -> BaseTool:
    """Factory：生成 delegate_to_subagent tool（Poirot self-copy subagent）。

    leaf role（INV#4）：子 agent tool_groups 不含 multiagent，不能 spawn。
    shared thread sandbox（INV#3）：复用父 sandbox_id。
    """

    @tool("delegate_to_subagent")
    def delegate_tool(
        goal: str,
        success_criteria: str,
        sandbox_id: str | None = None,
    ) -> str:
        """Delegate task to a Poirot self-copy subagent (leaf role, isolated context, shared sandbox). Provide goal and success_criteria."""
        state = get_current_state()
        resolved_sandbox_id = _extract_sandbox_id(state, sandbox_id)

        context_summary = context_summarizer.summarize(state, goal, success_criteria)

        request = SubagentRequest(
            goal=goal,
            success_criteria=success_criteria,
            context_summary=context_summary,
            sandbox_id=resolved_sandbox_id,
            artifacts_path=state.get("metadata", {}).get("artifacts_path"),
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        )

        try:
            sub_result = subagent_provider.spawn(request)
        except SubagentError as e:
            payload = {
                "success": False,
                "execution_status": "failed",
                "criteria_status": "unknown",
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
                "gap_analysis": str(e),
                "suggestion": "retry, fallback to specialist, or self-do",
            }
            if decision_log_writer is not None:
                _write_decision_log_async(
                    decision_log_writer,
                    "subagent",
                    goal,
                    success_criteria,
                    type("FailedResult", (), {"success": False, "failure_category": None})(),
                )
            return json.dumps(payload)

        evaluated = result_summarizer.summarize(
            sub_result.summary,
            list(sub_result.artifacts),
            goal,
            success_criteria,
        )

        if decision_log_writer is not None:
            _write_decision_log_async(
                decision_log_writer, "subagent", goal, success_criteria, evaluated
            )
        return json.dumps(_result_payload(evaluated, specialist_name="subagent"))

    return delegate_tool
