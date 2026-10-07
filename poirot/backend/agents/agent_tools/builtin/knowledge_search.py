"""Small local knowledge-base search tool for Agentic RAG.

The tool intentionally has no vector-store dependency.  It provides a local
BM25-style lexical index over Markdown/text files, while the Agent decides
whether to call it again with a refined query.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_SUPPORTED_SUFFIXES = frozenset({".md", ".txt", ".rst", ".json"})
_DEFAULT_ROOT = ".poirot/knowledge"
_DEFAULT_CHUNK_SIZE = 1200
_DEFAULT_CHUNK_OVERLAP = 180


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    path: Path
    title: str
    text: str
    tokens: tuple[str, ...]
    term_counts: dict[str, int]
    title_tokens: frozenset[str]


class LocalKnowledgeIndex:
    """Lazy, incrementally refreshed lexical index for local documents."""

    def __init__(
        self,
        root: Path,
        *,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.root = root
        self.chunk_size = max(200, chunk_size)
        self.chunk_overlap = min(max(0, chunk_overlap), self.chunk_size // 2)
        self._signatures: tuple[tuple[str, int, int], ...] = ()
        self._chunks: list[KnowledgeChunk] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0
        self._postings: dict[str, frozenset[int]] = {}

    def _files(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _SUPPORTED_SUFFIXES
            and not any(part.startswith(".") for part in path.relative_to(self.root).parts)
        )

    def _signature(self, path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        return (path.as_posix(), stat.st_mtime_ns, stat.st_size)

    def _split(self, path: Path, text: str) -> list[KnowledgeChunk]:
        title = path.stem
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip() or title
                break

        chunks: list[KnowledgeChunk] = []
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                digest = hashlib.sha1(
                    f"{path.as_posix()}:{chunk_index}:{start}".encode("utf-8")
                ).hexdigest()[:16]
                tokens = tuple(_tokenize(chunk_text))
                chunks.append(KnowledgeChunk(
                    chunk_id=f"kb-{digest}",
                    path=path,
                    title=title,
                    text=chunk_text,
                    tokens=tokens,
                    term_counts=dict(Counter(tokens)),
                    title_tokens=frozenset(_tokenize(title)),
                ))
                chunk_index += 1
            if end >= len(text):
                break
            start = max(start + 1, end - self.chunk_overlap)
        return chunks

    def refresh(self) -> None:
        files = self._files()
        signatures = tuple(self._signature(path) for path in files)
        if signatures == self._signatures:
            return

        chunks: list[KnowledgeChunk] = []
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            chunks.extend(self._split(path, text))

        document_frequency: Counter[str] = Counter()
        total_length = 0
        postings: dict[str, set[int]] = {}
        for chunk_index, chunk in enumerate(chunks):
            document_frequency.update(set(chunk.tokens))
            total_length += len(chunk.tokens)
            for token in chunk.term_counts:
                postings.setdefault(token, set()).add(chunk_index)

        self._signatures = signatures
        self._chunks = chunks
        self._document_frequency = document_frequency
        self._average_length = total_length / len(chunks) if chunks else 0.0
        self._postings = {
            token: frozenset(chunk_ids) for token, chunk_ids in postings.items()
        }

    def _score(self, query_tokens: list[str], chunk: KnowledgeChunk) -> float:
        if not chunk.tokens or not self._chunks or not self._average_length:
            return 0.0
        document_count = len(self._chunks)
        k1 = 1.5
        b = 0.75
        score = 0.0
        for token in query_tokens:
            frequency = chunk.term_counts.get(token, 0)
            if not frequency:
                continue
            df = self._document_frequency.get(token, 0)
            inverse_frequency = math.log(
                1.0 + (document_count - df + 0.5) / (df + 0.5)
            )
            length_norm = 1.0 - b + b * len(chunk.tokens) / self._average_length
            score += inverse_frequency * (
                frequency * (k1 + 1.0) / (frequency + k1 * length_norm)
            )

        score += 0.15 * len(set(query_tokens) & chunk.title_tokens)
        return score

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        self.refresh()
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        candidate_ids: set[int] = set()
        for token in query_tokens:
            candidate_ids.update(self._postings.get(token, ()))
        if not candidate_ids:
            return []

        scored = [
            (self._score(query_tokens, self._chunks[index]), self._chunks[index])
            for index in candidate_ids
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1].path.as_posix(), item[1].chunk_id))

        results: list[dict[str, Any]] = []
        for score, chunk in scored[: max(1, min(max_results, 8))]:
            absolute_path = chunk.path.resolve()
            results.append({
                "source_id": chunk.chunk_id,
                "title": chunk.title,
                "path": chunk.path.relative_to(self.root).as_posix(),
                "source_url": absolute_path.as_uri(),
                "snippet": chunk.text[:800],
                "score": round(score, 4),
            })
        return results


_INDEXES: dict[str, LocalKnowledgeIndex] = {}
_INDEX_LOCK = threading.Lock()


def _knowledge_root() -> Path:
    configured = os.environ.get("POIROT_RAG_DIR", _DEFAULT_ROOT).strip()
    return Path(configured or _DEFAULT_ROOT).expanduser()


def _get_index() -> LocalKnowledgeIndex:
    root = _knowledge_root().resolve()
    key = root.as_posix()
    with _INDEX_LOCK:
        index = _INDEXES.get(key)
        if index is None:
            index = LocalKnowledgeIndex(root)
            _INDEXES[key] = index
        return index


def reset_knowledge_index() -> None:
    """Clear cached indexes for tests and development reloads."""
    with _INDEX_LOCK:
        _INDEXES.clear()


@tool("knowledge_search")
def knowledge_search(query: str, max_results: int = 5) -> str:
    """Search local project knowledge and return source-grounded snippets.

    Use this tool when the task needs information from documents configured in
    POIROT_RAG_DIR.  Refine the query and call again when the evidence is not
    sufficient; do not treat an empty result as evidence that a fact is false.
    """
    try:
        root = _knowledge_root()
        results = _get_index().search(query, max_results=max_results)
        payload: dict[str, Any] = {
            "query": query,
            "knowledge_root": str(root),
            "total_results": len(results),
            "results": results,
        }
        if not results:
            payload["message"] = (
                "No matching local knowledge was found. "
                "The knowledge directory may be empty or the query may need refinement."
            )
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - defensive tool boundary
        return json.dumps({
            "error": f"Knowledge search failed: {exc}",
            "query": query,
        }, ensure_ascii=False)
