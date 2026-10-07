"""Tavily search provider for the unified web search tool."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_TAVILY_ENDPOINT = "https://api.tavily.com/search"
TAVILY_TIMEOUT_SECONDS = 15


def search_tavily(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search Tavily and return the internal normalized result shape."""
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured")

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    request = Request(
        os.environ.get("TAVILY_API_URL", DEFAULT_TAVILY_ENDPOINT),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=TAVILY_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid JSON response") from exc

    if not isinstance(data, dict):
        raise RuntimeError("invalid response shape")
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError("response did not contain results")

    normalized = [
        {
            "title": str(result.get("title", "")),
            "url": str(result.get("url", "")),
            "content": str(result.get("content", "")),
        }
        for result in raw_results
        if isinstance(result, dict)
    ]
    if not normalized:
        raise RuntimeError("No results found")
    return normalized
