"""Runtime tests for configured MCP fallback chains."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool

from poirot.backend.agents.mcp.audit import McpAuditMiddleware
from poirot.backend.agents.mcp.config import McpConfig
from poirot.backend.agents.mcp.registry import ToolEntry, ToolRegistry


@tool("mcp_search")
def mcp_search(query: str) -> str:
    """Primary search tool."""
    return query


@tool("ddg_search")
def ddg_search(query: str) -> str:
    """Built-in fallback search tool."""
    return query


class _Journal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def append(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


class _Runtime:
    def __init__(self, journal: _Journal) -> None:
        self.context = {"journal": journal}


@dataclass
class _Request:
    tool_call: dict
    tool: BaseTool
    state: dict
    runtime: _Runtime

    def override(self, **overrides):
        return replace(self, **overrides)


def _setup():
    config = McpConfig(
        fallback_chains={
            "mcp_search": ["search_server:mcp_search", "builtin:ddg_search"],
        }
    )
    registry = ToolRegistry(config)
    primary = ToolEntry(tool=mcp_search, source="mcp", server_name="search_server")
    fallback = ToolEntry(tool=ddg_search, source="builtin")
    registry.register(primary)
    registry.register(fallback)
    journal = _Journal()
    request = _Request(
        tool_call={"name": "mcp_search", "id": "call-original", "args": {"query": "x"}},
        tool=mcp_search,
        state={},
        runtime=_Runtime(journal),
    )
    return McpAuditMiddleware(registry), primary, fallback, journal, request


def test_primary_mcp_success_does_not_use_fallback():
    middleware, _, _, journal, request = _setup()
    calls: list[str] = []

    def handler(current):
        calls.append(current.tool_call["name"])
        return ToolMessage(content="primary result", tool_call_id=current.tool_call["id"])

    result = middleware.wrap_tool_call(request, handler)

    assert result.content == "primary result"
    assert calls == ["mcp_search"]
    assert journal.events[-1][1]["status"] == "ok"


def test_sync_primary_failure_uses_builtin_fallback_and_preserves_call_id():
    middleware, primary, fallback, journal, request = _setup()
    calls: list[str] = []

    def handler(current):
        calls.append(current.tool_call["name"])
        if current.tool is mcp_search:
            raise ConnectionError("primary offline")
        assert current.tool is ddg_search
        assert current.tool_call["id"] == "call-original"
        return ToolMessage(content="fallback result", tool_call_id=current.tool_call["id"])

    result = middleware.wrap_tool_call(request, handler)

    assert result.content == "fallback result"
    assert result.tool_call_id == "call-original"
    assert calls == ["mcp_search", "ddg_search"]
    assert primary.breaker.failure_count == 1
    assert fallback.breaker.failure_count == 0
    assert [payload["status"] for _, payload in journal.events] == ["error", "fallback_ok"]
    assert journal.events[-1][1]["selected_tool"] == "ddg_search"


def test_open_circuit_skips_primary_and_uses_fallback():
    middleware, primary, _, journal, request = _setup()
    for _ in range(3):
        primary.breaker.record_failure()
    calls: list[str] = []

    def handler(current):
        calls.append(current.tool_call["name"])
        return ToolMessage(content="fallback result", tool_call_id=current.tool_call["id"])

    result = middleware.wrap_tool_call(request, handler)

    assert result.content == "fallback result"
    assert calls == ["ddg_search"]
    assert [payload["status"] for _, payload in journal.events] == ["circuit_open", "fallback_ok"]


def test_all_fallback_candidates_fail_returns_single_paired_error():
    middleware, _, _, journal, request = _setup()

    def handler(current):
        raise ConnectionError(f"{current.tool_call['name']} offline")

    result = middleware.wrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.tool_call_id == "call-original"
    assert len(journal.events) == 2
    assert all(payload["status"] == "error" for _, payload in journal.events)


def test_async_fallback_uses_same_chain():
    middleware, _, _, _, request = _setup()

    async def handler(current):
        if current.tool is mcp_search:
            raise ConnectionError("primary offline")
        return ToolMessage(content="async fallback", tool_call_id=current.tool_call["id"])

    result = asyncio.run(middleware.awrap_tool_call(request, handler))
    assert result.content == "async fallback"
    assert result.tool_call_id == "call-original"


def test_async_mcp_timeout_uses_fallback():
    middleware, primary, _, journal, request = _setup()
    primary.metadata["timeout"] = 0.01

    async def handler(current):
        if current.tool is mcp_search:
            await asyncio.sleep(0.05)
            return ToolMessage(content="late", tool_call_id=current.tool_call["id"])
        return ToolMessage(content="timeout fallback", tool_call_id=current.tool_call["id"])

    result = asyncio.run(middleware.awrap_tool_call(request, handler))
    assert result.content == "timeout fallback"
    assert journal.events[0][1]["status"] == "error"
    assert "timeout after 0.01s" in journal.events[0][1]["error"]
