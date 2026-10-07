"""SkillMetricsMiddleware — applied 打点 + completions/fallbacks 归因。

INVARIANT:
- awrap_tool_call: tool ∈ active skill.allowed_tools → _applied_ctx[sid]=True + journal skill.apply
- after_agent: 判 task_completed（run 级，与 skill 正交）+ 对 active skill 归因 record_outcome
  - tool-skill（有 allowed_tools）未命中工具 → applied=False
  - guidance-skill（无 allowed_tools）→ applied=None（不强判）
  - applied=None → 只 selections 已打，不归因 completion/fallback（INV#9）
- provenance 经 _ctx ContextVar 桥接（awrap_tool_call 无 state）
- 无 _applied_ctx（无 injection middleware）→ 降级，after_agent 跳过归因（§6.6）
- 无 store/journal 静默不报错（INV#13）
- aafter_agent 委托同步
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langgraph.runtime import Runtime

from poirot.backend.agents.middlewares.run_journal_middleware import _get_runtime_value
from poirot.backend.agents.skill._ctx import _active_skills_ctx, _applied_ctx


class SkillMetricsMiddleware(AgentMiddleware):
    """applied + completions/fallbacks 打点。provenance 贯穿。"""

    def __init__(self, store: Any, skill_manager: Any | None = None) -> None:
        self._store = store
        # SkillManager is deliberately retained as a facade instead of storing the
        # L2/L3 objects directly: bootstrap attaches those optional layers after
        # load_startup() has already created this middleware.
        self._skill_manager = skill_manager

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        tool_call = getattr(request, "tool_call", None) or {}
        tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else ""
        runtime = getattr(request, "runtime", None)

        active_list = _active_skills_ctx.get()
        applied_map = _applied_ctx.get()
        if active_list and applied_map is not None and tool_name:
            journal = _get_runtime_value(runtime, "journal", None)
            for sid, allowed_tools in active_list:
                if tool_name in allowed_tools:
                    applied_map[sid] = True
                    if journal is not None:
                        try:
                            journal.append("skill.apply", {
                                "skill_id": sid, "tool_name": tool_name,
                            })
                        except Exception:
                            pass
            _applied_ctx.set(applied_map)

        return await handler(request)

    def after_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        store = _get_runtime_value(runtime, "skill_store") or self._store
        if store is None:
            return None
        run_id = _get_runtime_value(runtime, "run_id", None)

        metadata = state.get("metadata") or {}
        active_ids = metadata.get("active_skills") or []
        if not active_ids:
            return None

        applied_map = _applied_ctx.get()
        task_completed = self._judge_task_completed(state)

        for sid in active_ids:
            try:
                rec = store.get(sid)
            except Exception:
                rec = None
            applied = applied_map.get(sid) if applied_map else None
            # tool-skill 有 allowed_tools 但未命中 → False（有机会用没用）
            if rec and rec.allowed_tools and applied is None:
                applied = False
            # guidance-skill（无 allowed_tools）→ applied 保持 None
            try:
                store.record_outcome(sid, run_id, applied, task_completed)
            except Exception:
                pass
        self._schedule_lifecycle(
            state=state,
            runtime=runtime,
            store=store,
            active_ids=active_ids,
            task_completed=task_completed,
            run_id=str(run_id or ""),
        )
        return None

    async def aafter_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)

    def _schedule_lifecycle(
        self,
        *,
        state: Any,
        runtime: Any,
        store: Any,
        active_ids: list[str],
        task_completed: bool,
        run_id: str,
    ) -> None:
        """Run L3/L2 after the response without extending the user-facing turn."""
        if self._skill_manager is None:
            return

        coroutine = self._run_lifecycle(
            state=state,
            runtime=runtime,
            store=store,
            active_ids=active_ids,
            task_completed=task_completed,
            run_id=run_id,
        )
        try:
            asyncio.get_running_loop().create_task(coroutine)
        except RuntimeError:
            # Synchronous graph invocation has no event loop. A daemon thread
            # preserves the same fire-and-forget contract as async invocation.
            threading.Thread(
                target=lambda: asyncio.run(coroutine),
                daemon=True,
                name="skill-lifecycle",
            ).start()

    async def _run_lifecycle(
        self,
        *,
        state: Any,
        runtime: Any,
        store: Any,
        active_ids: list[str],
        task_completed: bool,
        run_id: str,
    ) -> None:
        """Evaluate the completed task, then ask the evolution manager to react."""
        manager = self._skill_manager
        if manager is None:
            return

        metadata = state.get("metadata") or {}
        task_id = run_id or str(metadata.get("thread_id") or "skill-eval")
        journal = _get_runtime_value(runtime, "journal", None)
        messages_summary = self._messages_summary(state)
        injected_skills = self._injected_skills(store, active_ids)

        eval_layer = manager.get_eval_layer()
        suggestions: list[Any] = []
        if eval_layer is not None:
            analyzer = getattr(eval_layer, "judgment_analyzer", None)
            if analyzer is not None and injected_skills:
                try:
                    _, suggestions = await analyzer.analyze_execution(
                        task_id=task_id,
                        journal_events=self._journal_events(journal),
                        messages_summary=messages_summary,
                        injected_skills=injected_skills,
                        task_completed=task_completed,
                        # L1 already recorded this run synchronously above.
                        record_outcomes=False,
                    )
                except Exception:
                    suggestions = []

            judge = getattr(eval_layer, "task_judge", None)
            if judge is not None:
                try:
                    await judge.judge_task(
                        task_id=task_id,
                        execution_trace=messages_summary,
                        final_output=self._final_output(state),
                    )
                except Exception:
                    pass

        evolution = manager.get_evolution_manager()
        if evolution is None:
            return
        try:
            if suggestions and hasattr(evolution, "run_suggestions"):
                await asyncio.to_thread(evolution.run_suggestions, suggestions)
            # Metric triggers are responsible for threshold and cooldown checks,
            # so invoking a cycle after a task does not imply every task mutates a Skill.
            await asyncio.to_thread(evolution.run_cycle)
        except Exception:
            pass

    @staticmethod
    def _messages_summary(state: Any) -> str:
        messages = state.get("messages") or []
        lines: list[str] = []
        for message in messages[-50:]:
            role = getattr(message, "type", message.__class__.__name__)
            content = getattr(message, "content", "")
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _final_output(state: Any) -> str:
        report = state.get("final_report")
        if report:
            return str(report)
        for message in reversed(state.get("messages") or []):
            if getattr(message, "type", "") not in {"ai", "assistant"}:
                continue
            content = getattr(message, "content", "")
            return str(content)
        return ""

    @staticmethod
    def _injected_skills(store: Any, active_ids: list[str]) -> list[dict[str, Any]]:
        skills: list[dict[str, Any]] = []
        for skill_id in active_ids:
            try:
                record = store.get(skill_id)
            except Exception:
                record = None
            if record is None:
                continue
            skills.append({
                "skill_id": record.skill_id,
                "name": record.name,
                "description": record.description,
            })
        return skills

    @staticmethod
    def _journal_events(journal: Any) -> list[dict]:
        if journal is None:
            return []
        for attr in ("events", "read_events"):
            value = getattr(journal, attr, None)
            try:
                events = value() if callable(value) else value
            except Exception:
                continue
            if isinstance(events, list):
                normalized: list[dict] = []
                for event in events:
                    if isinstance(event, dict):
                        normalized.append(event)
                    elif (
                        isinstance(event, tuple)
                        and len(event) == 2
                        and isinstance(event[0], str)
                        and isinstance(event[1], dict)
                    ):
                        normalized.append({
                            "event_type": event[0],
                            "payload": event[1],
                        })
                return normalized

        events_path = getattr(journal, "events_path", None)
        if not events_path:
            return []
        try:
            # RunJournal writes pretty-printed JSON records separated by blank
            # lines, despite the historical .jsonl filename. Parse complete
            # records rather than individual physical lines.
            blocks = Path(events_path).read_text(encoding="utf-8").split("\n\n")
            return [
                event
                for block in blocks
                if block.strip()
                for event in [json.loads(block)]
                if isinstance(event, dict)
            ]
        except Exception:
            return []

    @staticmethod
    def _judge_task_completed(state: Any) -> bool:
        """run 级任务完成判定。近似代理信号，非 ground-truth（INV#8）。

        run 模式：final_report 生成 + 无硬失败。
        chat 模式（无 report）：无硬失败（宽松）。
        """
        errors = state.get("errors") or []
        hard_failures = [e for e in errors if not _is_success_error(e)]
        if state.get("final_report"):
            return len(hard_failures) == 0
        return len(hard_failures) == 0


def _is_success_error(e: Any) -> bool:
    """errors 条目是否为 success 类（非硬失败）。兼容 dict / AgentError dataclass。"""
    if isinstance(e, dict):
        return e.get("kind") == "success"
    return getattr(e, "kind", None) == "success"
