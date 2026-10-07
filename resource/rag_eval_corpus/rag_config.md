# Local Agentic RAG configuration

The lightweight local Agentic RAG uses the knowledge_search builtin tool.
POIROT_RAG_DIR selects the local knowledge directory. Documents with md, txt,
rst, and json suffixes are loaded lazily, split into overlapping chunks, and
ranked by a local BM25-style lexical index.

The Agent decides whether to call knowledge_search, can refine the query, and
can call it again when the first evidence is insufficient. This version does
not use Chroma, Redis, embeddings, or a reranker.
