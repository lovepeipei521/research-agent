from __future__ import annotations

import json
from importlib import import_module

from poirot.backend.agents.agent_tools.builtin import get_builtin_tools
from poirot.backend.agents.agent_tools.available import _tool_group

knowledge_module = import_module(
    "poirot.backend.agents.agent_tools.builtin.knowledge_search"
)


def test_knowledge_search_returns_ranked_local_sources(tmp_path, monkeypatch) -> None:
    (tmp_path / "sandbox.md").write_text(
        "# Sandbox\nDocker sandbox isolates command execution from the host.",
        encoding="utf-8",
    )
    (tmp_path / "memory.md").write_text(
        "# Memory\nLong-term memory stores user preferences.",
        encoding="utf-8",
    )
    monkeypatch.setenv("POIROT_RAG_DIR", str(tmp_path))
    knowledge_module.reset_knowledge_index()

    result = json.loads(knowledge_module.knowledge_search.invoke({
        "query": "Docker sandbox command execution",
        "max_results": 3,
    }))

    assert result["total_results"] == 1
    assert result["results"][0]["path"] == "sandbox.md"
    assert result["results"][0]["source_url"].startswith("file://")


def test_knowledge_search_handles_empty_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("POIROT_RAG_DIR", str(tmp_path))
    knowledge_module.reset_knowledge_index()

    result = json.loads(knowledge_module.knowledge_search.invoke({"query": "missing"}))

    assert result["total_results"] == 0
    assert "message" in result


def test_knowledge_search_is_core_builtin_tool() -> None:
    names = {tool.name for tool in get_builtin_tools()}

    assert "knowledge_search" in names
    assert _tool_group("knowledge_search") == "core"
