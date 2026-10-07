"""MCP per-tool-call 审计中间件 — 熔断器 + 审计 + 外化联动标记。

INVARIANT:
- awrap_tool_call 拦截所有工具调用（MCP/builtin/sandbox 统一）
- 熔断器检查：open 不调用，记 circuit_open 事件
- 成功：record_success + 记 ok 事件
- 失败：record_failure + sanitize_error + 记 error 事件
- 外化联动：result 含 POIROT_EXTERNALIZED 标记时加 externalized=true + externalized_path
- 凭证脱敏：error_text 经 sanitizer 清洗后回 LLM
- 无 journal 时静默不记（不报错）
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import cast
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from poirot.backend.agents.middlewares.run_journal_middleware import _get_runtime_value
from poirot.backend.agents.middlewares.tagged_context_middleware import (
    POIROT_EXTERNALIZED,
    POIROT_EXTERNALIZED_META,
    POIROT_EXTERNALIZED_PATH,
)
from poirot.backend.agents.mcp.guards.credential_sanitizer import CredentialSanitizer
from poirot.backend.agents.mcp.registry import ToolRegistry

logger = logging.getLogger(__name__)

_BASH_OUTPUT_MAX_CHARS = 10000


def _truncate(text: str, max_chars: int = _BASH_OUTPUT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, {len(text) - max_chars} chars omitted)"


def _summarize_args(args: Any) -> str:
    """args 摘要（截断防日志爆炸）。"""
    try:
        import json
        s = json.dumps(args, ensure_ascii=False, default=str)
        return _truncate(s, 200)
    except Exception:
        return _truncate(str(args), 200)


class McpAuditMiddleware(AgentMiddleware):
    """per-tool-call 审计 + 熔断器联动。

    INVARIANT:
    - 拦截所有工具调用，统一记 tool.call 事件到 thread-events.jsonl
    - 熔断器 open 时不调用工具，记 circuit_open 事件
    - 凭证脱敏：错误信息经 CredentialSanitizer 清洗后回 LLM
    - 外化联动：result 含 POIROT_EXTERNALIZED 标记时加 externalized 字段
    - 无 journal 时静默不记（不报错）
    """

    def __init__(
        self,
        registry: ToolRegistry,
        sanitizer: CredentialSanitizer | None = None,
    ) -> None:
        self._registry = registry
        self._sanitizer = sanitizer or CredentialSanitizer()

    def _journal(self, runtime: Runtime) -> Any:
        return _get_runtime_value(runtime, "journal", None)

    def _emit(self, runtime: Runtime, event_type: str, payload: dict) -> None:
        journal = self._journal(runtime)
        if journal is not None:
            try:
                journal.append(event_type, payload)
            except Exception as exc:
                logger.debug("journal append failed: %s", exc)

    @staticmethod
    def _is_tool_error(result: Any) -> bool:
        """Only explicit tool errors trigger fallback; an empty result may be valid."""
        if isinstance(result, ToolMessage):
            return getattr(result, "status", None) == "error"
        if isinstance(result, Command):
            return any(
                McpAuditMiddleware._is_tool_error(message)
                for message in result.update.get("messages", [])
            )
        return False

    def _candidates(self, tool_name: str):
        """Resolve a primary MCP tool plus its explicitly configured fallbacks."""
        primary = self._registry.get(tool_name)
        if primary is None or primary.source != "mcp":
            return primary, [primary] if primary is not None else []
        entries = self._registry.get_fallback_entries(tool_name)
        if primary not in entries:
            entries.insert(0, primary)
        return primary, entries

    @staticmethod
    def _fallback_request(request: ToolCallRequest, entry: Any) -> ToolCallRequest:
        """Switch tool implementation but preserve the model's original call id."""
        tool_call = dict(request.tool_call)
        tool_call["name"] = entry.tool.name
        # ToolCallRequest.override accepts dataclass replacements at runtime;
        # ``tool`` is intentionally changed together with the tool-call name.
        return cast(
            ToolCallRequest,
            request.override(tool_call=tool_call, tool=entry.tool),  # type: ignore[call-arg]
        )

    def _audit_candidate(
        self,
        runtime: Runtime,
        original_name: str,
        entry: Any,
        tool_call: dict,
        result: Any,
        status: str,
        start: float,
        *,
        fallback_from: str | None = None,
        error: str | None = None,
    ) -> None:
        self._audit(
            runtime,
            original_name,
            entry.source,
            tool_call,
            result,
            status,
            start,
            error=error,
            selected_tool=entry.tool.name,
            fallback_from=fallback_from,
        )

    def _fallback_error_message(
        self,
        tool_call: dict,
        name: str,
        candidates: list[Any],
    ) -> ToolMessage:
        configured = self._registry.get_fallback_keys(name)
        if not configured:
            content = f"tool {name} failed; no fallback is configured"
        elif not candidates or len(candidates) < len(configured):
            content = (
                f"tool {name} has no usable configured fallback candidate "
                f"(configured: {', '.join(configured)})"
            )
        else:
            tried = ", ".join(entry.tool.name for entry in candidates)
            content = (
                f"tool {name} failed; all configured fallbacks failed "
                f"(tried: {tried})"
            )
        return ToolMessage(
            content=content,
            tool_call_id=tool_call.get("id", ""),
            status="error",
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "")
        runtime = request.runtime

        start = time.time()
        primary, candidates = self._candidates(tool_name)
        if primary is None or primary.source != "mcp":
            return await self._call_once_async(request, handler, primary, start)
        return await self._call_with_fallback_async(
            request, handler, candidates, start
        )

    async def _call_once_async(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
        entry: Any,
        start: float,
    ) -> Any:
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "")
        runtime = request.runtime
        source = entry.source if entry else "unknown"
        if entry and not entry.breaker.allow_call():
            self._audit(runtime, tool_name, source, tool_call, None, "circuit_open", start)
            return ToolMessage(
                content=f"tool {tool_name} unavailable (circuit open), try alternative",
                tool_call_id=tool_call.get("id", ""), status="error",
            )
        try:
            result = await self._invoke_async(handler, request, entry)
            if entry:
                entry.breaker.record_success()
            self._audit(runtime, tool_name, source, tool_call, result, "ok", start)
            return result
        except Exception as exc:
            if entry:
                entry.breaker.record_failure()
            self._audit(
                runtime, tool_name, source, tool_call, None, "error", start,
                error=self._sanitizer.sanitize_error(str(exc)),
            )
            raise

    async def _call_with_fallback_async(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
        candidates: list[Any],
        start: float,
    ) -> Any:
        original_name = request.tool_call.get("name", "")
        last_error: Exception | None = None
        primary_circuit_open = False
        for index, entry in enumerate(candidates):
            fallback_from = original_name if index else None
            if not entry.breaker.allow_call():
                primary_circuit_open = primary_circuit_open or index == 0
                self._audit_candidate(
                    request.runtime, original_name, entry, request.tool_call, None,
                    "circuit_open", start, fallback_from=fallback_from,
                )
                continue
            candidate_request = (
                request if index == 0 else self._fallback_request(request, entry)
            )
            try:
                result = await self._invoke_async(handler, candidate_request, entry)
            except Exception as exc:
                entry.breaker.record_failure()
                last_error = exc
                error_text = (
                    f"timeout after {entry.metadata.get('timeout')}s"
                    if isinstance(exc, asyncio.TimeoutError)
                    else self._sanitizer.sanitize_error(str(exc))
                )
                self._audit_candidate(
                    request.runtime, original_name, entry, request.tool_call, None,
                    "error", start, fallback_from=fallback_from,
                    error=error_text,
                )
                continue
            if self._is_tool_error(result):
                entry.breaker.record_failure()
                self._audit_candidate(
                    request.runtime, original_name, entry, request.tool_call, result,
                    "error", start, fallback_from=fallback_from,
                )
                continue
            entry.breaker.record_success()
            self._audit_candidate(
                request.runtime, original_name, entry, request.tool_call, result,
                "fallback_ok" if index else "ok", start,
                fallback_from=fallback_from,
            )
            return result
        if primary_circuit_open and len(candidates) == 1:
            return ToolMessage(
                content=f"tool {original_name} unavailable (circuit open), try alternative",
                tool_call_id=request.tool_call.get("id", ""), status="error",
            )
        if last_error is not None and len(candidates) == 1:
            raise last_error
        return self._fallback_error_message(request.tool_call, original_name, candidates)

    @staticmethod
    async def _invoke_async(
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
        request: ToolCallRequest,
        entry: Any,
    ) -> Any:
        """Invoke an MCP tool with its configured per-call deadline."""
        timeout = entry.metadata.get("timeout") if entry is not None else None
        if isinstance(timeout, (int, float)) and timeout > 0:
            return await asyncio.wait_for(handler(request), timeout=float(timeout))
        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Synchronous counterpart used by the TUI's graph.invoke path."""
        tool_name = request.tool_call.get("name", "")
        start = time.time()
        primary, candidates = self._candidates(tool_name)
        if primary is None or primary.source != "mcp":
            return self._call_once_sync(request, handler, primary, start)
        return self._call_with_fallback_sync(request, handler, candidates, start)

    def _call_once_sync(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
        entry: Any,
        start: float,
    ) -> Any:
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "")
        source = entry.source if entry else "unknown"
        if entry and not entry.breaker.allow_call():
            self._audit(request.runtime, tool_name, source, tool_call, None, "circuit_open", start)
            return ToolMessage(
                content=f"tool {tool_name} unavailable (circuit open), try alternative",
                tool_call_id=tool_call.get("id", ""), status="error",
            )
        try:
            result = handler(request)
            if entry:
                entry.breaker.record_success()
            self._audit(request.runtime, tool_name, source, tool_call, result, "ok", start)
            return result
        except Exception as exc:
            if entry:
                entry.breaker.record_failure()
            self._audit(
                request.runtime, tool_name, source, tool_call, None, "error", start,
                error=self._sanitizer.sanitize_error(str(exc)),
            )
            raise

    def _call_with_fallback_sync(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
        candidates: list[Any],
        start: float,
    ) -> Any:
        original_name = request.tool_call.get("name", "")
        last_error: Exception | None = None
        primary_circuit_open = False
        for index, entry in enumerate(candidates):
            fallback_from = original_name if index else None
            if not entry.breaker.allow_call():
                primary_circuit_open = primary_circuit_open or index == 0
                self._audit_candidate(
                    request.runtime, original_name, entry, request.tool_call, None,
                    "circuit_open", start, fallback_from=fallback_from,
                )
                continue
            candidate_request = (
                request if index == 0 else self._fallback_request(request, entry)
            )
            try:
                result = handler(candidate_request)
            except Exception as exc:
                entry.breaker.record_failure()
                last_error = exc
                self._audit_candidate(
                    request.runtime, original_name, entry, request.tool_call, None,
                    "error", start, fallback_from=fallback_from,
                    error=self._sanitizer.sanitize_error(str(exc)),
                )
                continue
            if self._is_tool_error(result):
                entry.breaker.record_failure()
                self._audit_candidate(
                    request.runtime, original_name, entry, request.tool_call, result,
                    "error", start, fallback_from=fallback_from,
                )
                continue
            entry.breaker.record_success()
            self._audit_candidate(
                request.runtime, original_name, entry, request.tool_call, result,
                "fallback_ok" if index else "ok", start,
                fallback_from=fallback_from,
            )
            return result
        if primary_circuit_open and len(candidates) == 1:
            return ToolMessage(
                content=f"tool {original_name} unavailable (circuit open), try alternative",
                tool_call_id=request.tool_call.get("id", ""), status="error",
            )
        if last_error is not None and len(candidates) == 1:
            raise last_error
        return self._fallback_error_message(request.tool_call, original_name, candidates)

    def _audit(
        self,
        runtime: Runtime,
        tool_name: str,
        source: str,
        tool_call: dict,
        result: Any,
        status: str,
        start: float,
        error: str | None = None,
        selected_tool: str | None = None,
        fallback_from: str | None = None,
    ) -> None:
        """写 tool.call 事件到 thread-events.jsonl。"""
        event: dict[str, Any] = {
            "tool_name": tool_name,
            "source": source,
            "args_summary": _summarize_args(tool_call.get("args")),
            "status": status,
            "duration_ms": int((time.time() - start) * 1000),
        }
        if error:
            event["error"] = error
        if selected_tool and selected_tool != tool_name:
            event["selected_tool"] = selected_tool
        if fallback_from:
            event["fallback_from"] = fallback_from
        # 外化联动标记
        if status == "ok" and isinstance(result, ToolMessage):
            additional_kwargs = getattr(result, "additional_kwargs", {}) or {}
            if additional_kwargs.get(POIROT_EXTERNALIZED):
                event["externalized"] = True
                event["externalized_path"] = additional_kwargs.get(POIROT_EXTERNALIZED_PATH, "")
                meta = additional_kwargs.get(POIROT_EXTERNALIZED_META) or {}
                if meta.get("externalized_ref"):
                    event["externalized_ref"] = meta["externalized_ref"]
        self._emit(runtime, "tool.call", event)
