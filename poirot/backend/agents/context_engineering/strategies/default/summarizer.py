from __future__ import annotations

import logging
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Mapping

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from poirot.backend.agents.context_engineering.contract import GovernanceResult
from poirot.backend.agents.context_engineering.strategies.default._constants import CST
from poirot.backend.agents.middlewares.tagged_context_middleware import (
    POIROT_EXTERNALIZED_META,
    POIROT_EXTERNALIZED_PATH,
    POIROT_SUMMARY,
)

if TYPE_CHECKING:
    from poirot.backend.agents.context_engineering.strategies.default.externalizer import ExternalizerExecutor

logger = logging.getLogger(__name__)


class SummarizerExecutor:
    """P4 全量 summarize + pairing 保护。"""

    def __init__(self, model: Any = None, preserve_recent: int = 6) -> None:
        self._model = model
        self._preserve_recent = preserve_recent

    def summarize_if_pending(
        self,
        governance: dict | None,
        messages: list,
        externalizer: ExternalizerExecutor,
        state: Mapping[str, Any] | None = None,
    ) -> GovernanceResult | None:
        governance = governance or {}
        pending = (governance.get("default") or {}).get("pending") or []
        if "P4" not in pending:
            return None
        if not self._model:
            return None
        to_summarize, preserved = self._partition(messages)
        if not to_summarize:
            return None
        externalized_paths = self._externalize_orphans(to_summarize, externalizer)
        summary_text = self._call_llm(to_summarize, state=state)
        # A failed/empty summary must never be followed by REMOVE_ALL_MESSAGES.
        # Keeping the original state is safer than replacing it with a placeholder.
        if not summary_text:
            logger.warning("P4 summary unavailable; preserving original message history")
            return None
        if externalized_paths:
            summary_text += (
                "\n\n外化工具结果：已由运行时保留 %d 个结果，当前上下文仅含预览。"
                % len(externalized_paths)
            )
        summary_msg = HumanMessage(content=summary_text, additional_kwargs={POIROT_SUMMARY: True})
        return GovernanceResult(
            state_patch={"governance": self._update_summary(governance, summary_text)},
            messages_patch=[RemoveMessage(id=REMOVE_ALL_MESSAGES), summary_msg, *preserved],
        )

    def _partition(self, messages: list) -> tuple[list, list]:
        n = len(messages)
        if n <= self._preserve_recent:
            return [], messages
        cut = n - self._preserve_recent
        cut = self._snap_to_pairing(messages, cut)
        to_summarize = list(messages[:cut])
        preserved = list(messages[cut:])
        # preserved 孤立 ToolMessage（配对 AIMessage 在 to_summarize）移到 to_summarize，防 400
        preserved, orphans = self._strip_orphan_tools(preserved)
        to_summarize.extend(orphans)
        return to_summarize, preserved

    @staticmethod
    def _strip_orphan_tools(preserved: list) -> tuple[list, list]:
        """preserved 中孤立 ToolMessage 或孤立 AIMessage(tool_calls) 移除，防 pairing 断裂。"""
        ai_tc_ids: set[str] = set()
        for msg in preserved:
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls or []:
                    tc_id = tc.get("id") if isinstance(tc, dict) else None
                    if tc_id:
                        ai_tc_ids.add(tc_id)
        tool_ids: set[str] = set()
        for msg in preserved:
            if isinstance(msg, ToolMessage):
                tool_ids.add(msg.tool_call_id)
        clean: list = []
        orphans: list = []
        for msg in preserved:
            is_orphan_tool = isinstance(msg, ToolMessage) and msg.tool_call_id not in ai_tc_ids
            is_orphan_ai = isinstance(msg, AIMessage) and bool(msg.tool_calls) and not all(
                (tc.get("id") if isinstance(tc, dict) else None) in tool_ids
                for tc in msg.tool_calls
            )
            if is_orphan_tool or is_orphan_ai:
                orphans.append(msg)
            else:
                clean.append(msg)
        return clean, orphans

    def _snap_to_pairing(self, messages: list, cut: int) -> int:
        while cut < len(messages) and isinstance(messages[cut], ToolMessage):
            if cut > 0 and isinstance(messages[cut - 1], AIMessage) and messages[cut - 1].tool_calls:
                cut -= 1
            else:
                break
        return cut

    def _externalize_orphans(self, messages: list, externalizer: ExternalizerExecutor) -> list[str]:
        ai_tc_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls or []:
                    tc_id = tc.get("id") if isinstance(tc, dict) else None
                    if tc_id:
                        ai_tc_ids.add(tc_id)
        paths: list[str] = []
        for index, msg in enumerate(messages):
            if isinstance(msg, ToolMessage) and msg.tool_call_id not in ai_tc_ids:
                rewritten = externalizer.externalize_if_needed(msg)
                if rewritten is not None:
                    messages[index] = rewritten
                    path = rewritten.additional_kwargs.get(POIROT_EXTERNALIZED_PATH)
                    if path:
                        paths.append(path)
        return paths

    def _call_llm(self, messages: list, state: Mapping[str, Any] | None = None) -> str | None:
        if not self._model:
            return None
        try:
            from poirot.backend.agents.prompts import get_prompt_manager

            history = self._format_history(messages)
            if not history.strip():
                return None
            prompt = get_prompt_manager().load(
                "context_engineering/default",
                "summarize",
                messages_text=history,
                state_context=self._format_state(state),
            )
            # config tags 标记内部调用，防 astream(stream_mode=messages) 捕获泄漏到 CLI
            response = self._model.invoke(prompt, config={"tags": ["internal_llm"]})
            text = response.text if hasattr(response, "text") else str(response)
            return text.strip() or None
        except Exception as exc:
            logger.warning("P4 summary model call failed: %s", exc)
            return None

    @staticmethod
    def _format_state(state: Mapping[str, Any] | None) -> str:
        """Keep structured research state available even if its original turn was removed."""
        if not state:
            return "(no structured state provided)"
        lines: list[str] = []
        question = state.get("research_question")
        if question:
            lines.append(f"research_question: {question}")
        todos = state.get("todos") or []
        if todos:
            lines.append("todos: " + str(todos))
        reflections = state.get("reflection_items") or []
        if reflections:
            lines.append("reflection_items: " + str(reflections))
        for field_name, limit in (("sources", 20), ("observations", 12), ("citations", 12)):
            records = state.get(field_name) or []
            if not records:
                continue
            lines.append(f"{field_name}:")
            for record in list(records)[-limit:]:
                if isinstance(record, Mapping):
                    value = json.dumps(record, ensure_ascii=False, default=str)
                else:
                    value = str(record)
                lines.append(f"- {value}")
        return "\n".join(lines) or "(no structured state provided)"

    @staticmethod
    def _format_history(messages: list) -> str:
        lines: list[str] = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            metadata: list[str] = []
            msg_id = getattr(msg, "id", None)
            if msg_id:
                metadata.append(f"id={msg_id}")
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id:
                metadata.append(f"tool_call_id={tool_call_id}")
            if isinstance(msg, AIMessage) and msg.tool_calls:
                call_ids = [
                    tc.get("id")
                    for tc in msg.tool_calls
                    if isinstance(tc, dict) and tc.get("id")
                ]
                if call_ids:
                    metadata.append("tool_call_ids=" + ",".join(call_ids))
            if isinstance(msg, ToolMessage):
                meta = msg.additional_kwargs.get(POIROT_EXTERNALIZED_META) or {}
                externalized_ref = meta.get("externalized_ref")
                if externalized_ref:
                    metadata.append(f"externalized_ref={externalized_ref}")
            suffix = (" " + " ".join(metadata)) if metadata else ""
            lines.append(f"[{type(msg).__name__}{suffix}] {content[:500]}")
        return "\n".join(lines)

    def _update_summary(self, governance: dict, summary_text: str) -> dict:
        g = dict(governance or {})
        d = dict(g.get("default") or {})
        d["summary"] = summary_text
        d["summary_id"] = "summary_" + datetime.now(CST).strftime("%H%M%S%f")
        metrics = dict(d.get("metrics") or {})
        metrics["summarize_count"] = metrics.get("summarize_count", 0) + 1
        d["metrics"] = metrics
        g["default"] = d
        return g
