from __future__ import annotations

import json

from poirot.backend.agents.agent_tools.builtin import web_search as web_search_module


def _tool_result(raw: str) -> dict:
    return json.loads(raw)


def test_auto_without_tavily_key_uses_ddgs(monkeypatch) -> None:
    monkeypatch.delenv("POIROT_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    calls: list[str] = []

    def fake_ddgs(query: str, max_results: int) -> list[dict[str, str]]:
        calls.append(f"ddgs:{query}:{max_results}")
        return [{"title": "DDG", "url": "https://example.com/ddg", "content": "fallback"}]

    monkeypatch.setattr(web_search_module, "search_ddg", fake_ddgs)
    monkeypatch.setattr(
        web_search_module,
        "search_tavily",
        lambda query, max_results: (_ for _ in ()).throw(AssertionError("Tavily should not run")),
    )

    result = _tool_result(web_search_module.web_search_tool.invoke({"query": "test", "max_results": 1}))

    assert calls == ["ddgs:test:1"]
    assert result["total_results"] == 1
    assert result["results"][0]["url"] == "https://example.com/ddg"


def test_auto_prefers_tavily_when_key_is_configured(monkeypatch) -> None:
    monkeypatch.setenv("POIROT_SEARCH_PROVIDER", "auto")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    calls: list[str] = []

    def fake_tavily(query: str, max_results: int) -> list[dict[str, str]]:
        calls.append("tavily")
        return [{"title": "Tavily", "url": "https://example.com/tavily", "content": "primary"}]

    monkeypatch.setattr(web_search_module, "search_tavily", fake_tavily)
    monkeypatch.setattr(
        web_search_module,
        "search_ddg",
        lambda query, max_results: (_ for _ in ()).throw(AssertionError("DDG should not run")),
    )

    result = _tool_result(web_search_module.web_search_tool.invoke({"query": "test"}))

    assert calls == ["tavily"]
    assert result["results"][0]["title"] == "Tavily"


def test_tavily_failure_falls_back_to_ddgs(monkeypatch) -> None:
    monkeypatch.setenv("POIROT_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    calls: list[str] = []

    def failing_tavily(query: str, max_results: int) -> list[dict[str, str]]:
        calls.append("tavily")
        raise RuntimeError("temporary outage")

    def fake_ddgs(query: str, max_results: int) -> list[dict[str, str]]:
        calls.append("ddgs")
        return [{"title": "Fallback", "url": "https://example.com/fallback", "content": "backup"}]

    monkeypatch.setattr(web_search_module, "search_tavily", failing_tavily)
    monkeypatch.setattr(web_search_module, "search_ddg", fake_ddgs)

    result = _tool_result(web_search_module.web_search_tool.invoke({"query": "test"}))

    assert calls == ["tavily", "ddgs"]
    assert result["results"][0]["title"] == "Fallback"
