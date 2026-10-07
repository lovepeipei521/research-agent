# RAG Evaluation V1

## Scope

This evaluation validates the lightweight local RAG implementation only:

```text
local Markdown corpus
-> chunking
-> BM25-style lexical retrieval
-> knowledge_search tool
-> Agent tool-selection and response checks
```

It does not measure general web search, vector retrieval, Chroma, embeddings,
rerankers, Redis, multi-agent collaboration, or production traffic.

## Retrieval Set

`poirot_rag_eval_v1.json` contains 20 fixed retrieval cases over
`rag_eval_corpus/`:

- direct fact lookup;
- platform and configuration lookup;
- memory, Skill, MCP, and context-governance lookup;
- negative queries with no matching corpus content;
- one Chinese semantic-paraphrase stress case.

The retrieval runner reports `Recall@k`, required-term coverage, negative
no-result rate, per-case pass rate, and mean tool latency.

## Agent Set

`poirot_agentic_rag_eval_v1.json` contains 20 fixed Agent tasks. Each task
defines whether `knowledge_search` should be called, expected document sources,
required response facts, forbidden claims, and optionally a minimum number of
distinct retrieval queries.

The Agent runner reports:

- tool-selection rate;
- evidence-retrieval rate from `tool.finished` journal events;
- response-contract rate;
- query-refinement rate for retry cases;
- completion rate, average tool calls, and latency;
- runtime errors separately from scored Agent cases.

## Evidence Boundary

Only metrics in a generated `summary.json` are reportable. Each summary keeps
the corpus path, case details, output traces, final response, and model
provider. A connection error is evaluation evidence, not an Agent failure.
