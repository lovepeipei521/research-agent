from pathlib import Path

import pytest

from poirot.backend.agents.agent_tools.builtin.knowledge_notes import save_learning_note


def test_learning_note_requires_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit confirmation"):
        save_learning_note("内容", root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_learning_note_is_markdown_and_refreshes_index(tmp_path: Path) -> None:
    path = save_learning_note(
        "BM25 检索学习总结",
        title="Agentic RAG / 入门",
        confirmed=True,
        root=tmp_path,
    )
    assert path.parent == tmp_path
    assert path.suffix == ".md"
    text = path.read_text(encoding="utf-8")
    assert "source: user-confirmed-learning-note" in text
    assert "BM25 检索学习总结" in text


def test_learning_note_rejects_empty_content(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        save_learning_note("  ", confirmed=True, root=tmp_path)
