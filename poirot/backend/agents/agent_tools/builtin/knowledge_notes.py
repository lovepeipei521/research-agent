"""User-confirmed learning-note persistence for the lightweight local RAG."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from poirot.backend.agents.agent_tools.builtin.knowledge_search import (
    _knowledge_root,
    reset_knowledge_index,
)

_DEFAULT_TITLE = "Learning note"
_SAFE_TITLE_RE = re.compile(r"[^0-9A-Za-z_\u4e00-\u9fff.-]+")


def _safe_title(title: str) -> str:
    cleaned = _SAFE_TITLE_RE.sub("-", title.strip()).strip("-._")
    return cleaned[:80] or _DEFAULT_TITLE.replace(" ", "-")


def save_learning_note(
    content: str,
    *,
    title: str = _DEFAULT_TITLE,
    confirmed: bool = False,
    root: Path | None = None,
) -> Path:
    """Save a learning note only after an explicit user confirmation.

    ``confirmed`` is intentionally required by callers; the CLI's ``/learn save``
    command is the user-facing confirmation point.  Notes are ordinary Markdown
    files so the existing local lexical index can discover them on the next query.
    """
    if not confirmed:
        raise ValueError("saving a learning note requires explicit confirmation")
    if not content.strip():
        raise ValueError("learning note is empty")

    knowledge_root = (root or _knowledge_root()).expanduser()
    knowledge_root.mkdir(parents=True, exist_ok=True)
    display_title = " ".join(title.split())[:120] or _DEFAULT_TITLE
    safe_title = _safe_title(display_title)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = knowledge_root / f"{timestamp}-{safe_title}.md"
    body = (
        "---\n"
        f"title: {display_title}\n"
        "source: user-confirmed-learning-note\n"
        f"created_at: {timestamp}\n"
        "---\n\n"
        f"# {display_title}\n\n"
        f"{content.strip()}\n"
    )
    path.write_text(body, encoding="utf-8")
    reset_knowledge_index()
    return path
