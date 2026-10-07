from langchain_core.messages import ToolMessage

from poirot.backend.agents.middlewares.evidence_middleware import _extract_sources


def test_page_fetch_records_requested_url_only() -> None:
    message = ToolMessage(
        content=(
            "Fetched https://docs.python.org/3/ . Navigation: "
            "https://example.com/unrelated"
        ),
        tool_call_id="call-1",
    )

    sources = _extract_sources(
        "browse_page",
        message,
        {"name": "browse_page", "args": {"url": "https://docs.python.org/3/"}},
    )

    assert [source.url for source in sources] == ["https://docs.python.org/3/"]


def test_parallel_page_fetch_records_each_requested_url() -> None:
    message = ToolMessage(
        content="Fetched two pages and linked https://example.com/noise",
        tool_call_id="call-2",
    )

    sources = _extract_sources(
        "parallel_browse",
        message,
        {
            "name": "parallel_browse",
            "args": {"urls": ["https://a.test", "https://b.test"]},
        },
    )

    assert [source.url for source in sources] == ["https://a.test", "https://b.test"]


def test_search_still_extracts_result_urls() -> None:
    message = ToolMessage(
        content="Result one https://a.test and result two https://b.test",
        tool_call_id="call-3",
    )

    sources = _extract_sources("web_search", message)

    assert [source.url for source in sources] == ["https://a.test", "https://b.test"]
