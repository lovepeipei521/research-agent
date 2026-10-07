"""Unified web search tool with configurable provider fallback."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable

from langchain_core.tools import tool

from poirot.backend.agents.agent_tools.builtin.ddg_search import search_ddg
from poirot.backend.agents.agent_tools.builtin.tavily_search import search_tavily

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 5
MAX_PROVIDER_ATTEMPTS = 2
Provider = Callable[[str, int], list[dict[str, str]]]


def _provider_order() -> list[tuple[str, Provider]]:
    provider = os.environ.get("POIROT_SEARCH_PROVIDER", "auto").strip().lower() or "auto"
    tavily_configured = bool(os.environ.get("TAVILY_API_KEY", "").strip())

    if provider == "ddgs":
        return [("ddgs", search_ddg)]
    if provider == "tavily":
        return [("tavily", search_tavily), ("ddgs", search_ddg)]
    if provider == "auto":
        if tavily_configured:
            return [("tavily", search_tavily), ("ddgs", search_ddg)]
        return [("ddgs", search_ddg)]
    raise ValueError(f"unsupported search provider: {provider}")


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    """Search the web for information. Use this tool to find current information, news, articles, and facts from the internet.

    Args:
        query: Search keywords describing what you want to find. Be specific for better results.
        max_results: Maximum number of results to return. Default is 5.
    """
    try:
        providers = _provider_order()
    except ValueError as exc:
        return json.dumps({"error": str(exc), "query": query}, ensure_ascii=False)

    failures: list[str] = []
    for name, provider in providers[:MAX_PROVIDER_ATTEMPTS]:
        try:
            results = provider(query, max_results)
            if not results:
                raise RuntimeError("No results found")
            return json.dumps(
                {
                    "query": query,
                    "provider": name,
                    "total_results": len(results),
                    "results": results,
                },
                indent=2,
                ensure_ascii=False,
            )
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            logger.warning("web search provider %s failed: %s", name, exc)

    return json.dumps(
        {
            "error": f"Search failed: {'; '.join(failures)}",
            "query": query,
            "providers_tried": [name for name, _ in providers[:MAX_PROVIDER_ATTEMPTS]],
        },
        ensure_ascii=False,
    )
