"""DuckDuckGo search provider used by the unified web search tool."""

from __future__ import annotations

import os

DEFAULT_MAX_RESULTS = 5
DEFAULT_REGION = "wt-wt"
DEFAULT_SAFESEARCH = "moderate"
DEFAULT_BACKEND = "auto"
DDGS_TIMEOUT_SECONDS = 15


def search_ddg(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    """Search DuckDuckGo and return the internal normalized result shape."""
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("ddgs library not installed. Run: pip install ddgs") from exc

    backend = os.environ.get("POIROT_SEARCH_DDGS_BACKEND", DEFAULT_BACKEND).strip()
    backend = backend or DEFAULT_BACKEND
    ddgs = DDGS(timeout=DDGS_TIMEOUT_SECONDS)
    try:
        results = ddgs.text(
            query,
            region=DEFAULT_REGION,
            safesearch=DEFAULT_SAFESEARCH,
            max_results=max_results,
            backend=backend,
        )
        raw_results = list(results) if results else []
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    normalized = [
        {
            "title": str(result.get("title", "")),
            "url": str(result.get("href", result.get("link", ""))),
            "content": str(result.get("body", result.get("snippet", ""))),
        }
        for result in raw_results
        if isinstance(result, dict)
    ]
    if not normalized:
        raise RuntimeError("No results found")
    return normalized
