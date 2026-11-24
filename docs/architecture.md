# Architecture

How the service is put together and, where it matters, why a design choice was made and
what alternative was rejected.

Start with the [Overview](#overview) if you are new to the codebase; the rest of this
document fills in the reasoning behind each decision.

## Overview

This is a question-answering service over a corpus of PDFs. You add documents (the
**write path**), then ask questions about them (the **read path**). Every answer cites the
document and page it came from, and a multi-tenant layer keeps each customer's corpus
isolated from every other's.

At runtime it is a small set of containers:

```
                 browser
                    │  HTTPS
                    ▼
        frontend (Caddy)      static SPA + reverse proxy for /api/*
                    │
                    ▼
        api (FastAPI)         auth · rate limit · the RAG pipeline
      ┌──────┼──────┬─────────────┐
      ▼      ▼      ▼             ▼
   qdrant  redis  postgres    Groq / Cohere
  (vectors) (cache, (tenants, (generation,
            limits,  users,    reranking)
            broker)  text index)
```

| Service | Job (plain English) |
|---|---|
| frontend | The website; also forwards `/api/*` to the API. |
| api | The "brain": authentication, the retrieval pipeline, streaming answers. |
| qdrant | "Meaning" search — stores vector embeddings and finds similar chunks. |
| redis | Remembers past answers (semantic cache) and counts rate limits. |
| postgres | Users, tenants, and the keyword (full-text) index. |
| worker | (optional) Background job that processes PDF uploads. |

The two flows in one line each:

- **Write path (ingestion):** PDF → parse → split into chunks → embed + index (Qdrant for
  vectors, Postgres for keyword text).
- **Read path (query):** question → hybrid recall (keyword + vector) → rerank → generate an
  answer with citations.

That diagram is enough to understand the *what*; the rest of this document explains the
*why*.

## Components

| Component | Technology | Responsibility |
|---|---|---|
| API | FastAPI + uvicorn | HTTP surface, auth, rate limiting, validation, streaming |
| Pipeline | `src/core/rag_pipeline.py` | Retrieve → rerank → generate, cache integration |
| Vector store | Qdrant | Child-chunk embeddings, payload-indexed by source/page |
| Keyword index | Postgres FTS (`tsvector`) or `rank_bm25` pickle | Lexical recall over the same child chunks |
| Document store | Postgres or `LocalFileStore` | Parent sections, addressed by `doc_id` |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d) | Local, CPU, no per-query API cost |
| Reranker | Cohere `rerank-v3.5` | Cross-encoder rescoring of the fused candidate set |
| Generator | Groq `openai/gpt-oss-120b` | Grounded answer generation |
| Cache | Redis Stack (HNSW vector index) | Semantic response cache + rate-limit counters |
| Task queue | Celery (Redis broker) | Async PDF ingestion for uploads |
| Frontend | React 19 + Vite, served by Caddy | Chat UI, SSE streaming, citation display |

Two pipelines exist and never run concurrently:

- **Ingestion** — offline, batch, writes the indexes (`scripts/ingest.py`, or Celery for
  uploads).
- **Query** — online, per-request, reads the indexes.

Both share the same splitter and embedding factories, because if they disagreed about how
text becomes chunks, retrieval would be comparing incomparable units.

## Code map

Where each concept lives, for when you need to find the implementation.

| Concept | Primary file(s) |
|---|---|
| App wiring / entrypoint | `src/api/main.py` |
| HTTP endpoints | `src/api/routes.py` |
| Auth, rate limiting, tenant context | `src/api/deps.py`, `src/api/jwt_auth.py` |
| Settings + production guard | `src/config/settings.py`, `src/config/production_guard.py` |
| End-to-end query orchestration | `src/core/rag_pipeline.py` |
| Prompt construction | `src/core/prompt_builder.py`, `src/core/prompts/*.json` |
| Query / faithfulness guards | `src/core/query_guard.py`, `src/core/faithfulness_guard.py` |
| PDF parsing, chunking | `src/ingestion/deepdoc_loader.py`, `src/ingestion/chunker.py` |
| Async upload task | `src/ingestion/tasks.py` |
| Vector store + embeddings | `src/retrieval/vector_store.py` |
| Keyword index | `src/db/pg_bm25.py` (Postgres) or `src/retrieval/bm25.py` (pickle) |
| Hybrid fusion | `src/retrieval/hybrid_search.py` |
| Reranker / generator | `src/llm/reranker.py`, `src/llm/generator.py` |
| Semantic cache | `src/cache/semantic_cache.py` |
| Database schema | `src/db/schema.py` |
| Observability | `src/observability/` |
| Ingestion CLI | `scripts/ingest.py` |
| Tenant/user CLI | `scripts/manage_tenants.py` |

Suggested reading order for a newcomer: `src/api/main.py` (wiring) → `src/api/routes.py`
(endpoints) → `src/core/rag_pipeline.py` (the pipeline) → `src/retrieval/hybrid_search.py`
(retrieval) → `src/llm/generator.py` (generation).

## Request lifecycle

```
RequestContextMiddleware      assign/propagate X-Request-ID, time the request
  └─ CORSMiddleware           explicit origin allow-list
      └─ authorize            verify JWT / X-API-Key, then charge the rate limit
          └─ get_pipeline     cached factory, scoped to the tenant
              └─ anyio.to_thread.run_sync(pipeline.run)
                  ├─ cache.get(query)
                  ├─ hybrid recall
                  ├─ resolve parents → dedupe → cap
                  ├─ reranker.rerank(...)
                  ├─ llm.answer(...)
                  └─ cache.set(query, answer, sources)
```

The pipeline runs in a worker thread because `pipeline.run` is synchronous and does CPU
work (embedding) plus blocking network I/O (Cohere, Groq). Calling it directly from an
`async def` handler would block the event loop for the duration of an answer.

## Ingestion

### Parsing

`DeepDocLoader` wraps `pymupdf4llm` with `page_chunks=True`, yielding one Markdown
`Document` per page with `source`, `page`, and `parser` metadata. `pymupdf4llm` applies
the same class of ONNX models as RAGFlow's DeepDoc: layout recognition to classify page
regions, table-structure recognition to convert tables to Markdown, and reading-order
reconstruction. If layout parsing fails, the loader falls back to raw PyMuPDF text
extraction rather than dropping the file.

*Rejected:* naive `page.get_text()` extraction — it destroys tables and flattens heading
structure, which are exactly the signals the chunker depends on.

### Chunking — parent and child

Two splits run per document:

- **Parent** — `MarkdownSectionSplitter` cuts on heading boundaries (`#`…`####`), with a
  paragraph-level fallback for oversized sections or documents with no headings. Parents
  are what the generator reads.
- **Child** — `DeepDocChildSplitter` cuts parents into ~700-character, 80-character-overlap
  windows. Children are what gets embedded and keyword-indexed.

*Why both:* a passage small enough to embed precisely is usually too small to answer from,
and a passage large enough to answer from embeds imprecisely. Small chunks are matched;
large parents are returned.

`split_documents` is used rather than `split_text`, because LangChain's document-level
split propagates metadata to every chunk. That is what puts `source` and `page` on a child
chunk and makes citations resolvable.

### Indexing and idempotency

The BM25 corpus and the dense index are derived from the same split, so a keyword hit and a
dense hit refer to identical text. Idempotency relies on:

1. **A content-addressed, target-aware manifest** — keyed by corpus-relative POSIX path,
   valued with the file's SHA-256, and stamped with the collection name and embedding model
   it was built for. Unchanged files are skipped, but only while the target is unchanged;
   a manifest with no recorded target is discarded rather than trusted.
2. **Delete-before-add** — re-indexing a file deletes its existing vectors and keyword
   chunks first, so edits do not leave stale passages retrievable.
3. **Atomic writes** — the manifest is written to a temp file and `os.replace`d, so an
   interrupted run cannot leave it half-written.

The manifest tracks file contents and target, not parsing/chunking code — after changing
`src/ingestion/`, re-run with `--reset`.

## Retrieval

### Hybrid recall

BM25 and dense results are fused with weighted reciprocal-rank fusion (`ENSEMBLE_WEIGHTS`,
default `0.4` keyword / `0.6` dense).

*Why not dense-only:* finance text is dense with exact tokens whose meaning embeddings blur
— ticker symbols, model names (`GARCH`, `Heston`), Greek letters, numeric parameter values.
Keyword recall catches what the embedding misses, and vice versa.

One canonical tokenizer (`[a-z0-9]+` over lowercased text) is shared by index construction
and querying so the two cannot drift. BM25 candidates are gated by token overlap with the
query rather than a score threshold, because `BM25Okapi` assigns negative IDFs to terms
appearing in most documents — a `score > 0` rule would silently discard valid matches in a
small corpus.

### Parent resolution

Dense retrieval returns parent sections, but BM25 indexes child chunks. Without
normalisation, fusion would mix units of two sizes. Each child carries the `doc_id` of its
parent, so `_resolve_parents` loads the parents from the document store and substitutes
them, carrying the retrieval scores across so citations still explain why a passage was
chosen.

### Deduplication and capping

Passages are deduplicated on `(source, page, sha1(content))`, then truncated to
`RERANK_MAX_CANDIDATES` (40). Multiple children of one parent collapse into that parent;
repeated boilerplate does not crowd out other results.

### Reranking

The fused candidate set is rescored by Cohere `rerank-v3.5` and truncated to
`top_k_rerank`. A bi-encoder compares embeddings independently; a cross-encoder reads query
and passage together, which is far more accurate and far too slow to run over the whole
corpus — which is why it runs last on a small set.

Two implementation details:

- Reranked documents are **copies**, not originals, because retrieved `Document` objects
  are shared with the retrieval layer and in-place metadata edits would leak across
  requests.
- Any reranker failure falls back to the fused ordering. Reranker trouble degrades answer
  quality; it does not cause an outage.

### Relevance floor

Passages scored below `RERANK_MIN_SCORE` are dropped before they can be cited, and if none
clear the floor the pipeline returns a fixed decline message with empty sources and **does
not call the LLM**. Retrieval always returns `k` results whether or not any are relevant;
without the floor, an unanswerable query would return an honest "I don't know" alongside a
list of near-miss citations that imply support that does not exist. The separation is wide:
a real match scores ~0.85–0.95, an off-corpus query below 0.1. The decline is deliberately
not cached, so it does not outlive a re-index that makes the question answerable.

## Generation and grounding

Context passages are labelled `[Source: <file>, page <n>]`, and the prompt instructs the
model to answer only from the context and to say so explicitly when it is insufficient.
Generation runs at `temperature=0.0` for reproducibility, with chain-of-thought reasoning
in `<thinking>` blocks that are stripped before the answer is returned.

### Faithfulness guard

Controlled by `ENABLE_FAITHFULNESS_GUARD` (default `true`), a second LLM call runs a
boolean entailment audit of the answer against the retrieved context:

| Path | Behaviour |
|------|----------|
| `POST /chat` | If the guard fails, the response is a safe refusal with empty sources. |
| `POST /chat/stream` | Sources and tokens are sent first; if the guard fails after buffering, an SSE `error` event is appended and `done` sets `faithfulness_passed: false`. |
| Semantic cache | `cache.set` runs only after the guard passes, so failed audits are never cached. |
| Guard errors | The pipeline fails open (allows the answer) and logs the failure. |

## Caching

`SemanticCache` embeds the incoming query and searches a Redis Stack HNSW index for
neighbours. A hit at or above `SEMANTIC_CACHE_THRESHOLD` (0.95) returns immediately,
skipping retrieval, reranking, and generation.

Every entry is written and read under a **scope** — a canonical string covering the tenant,
the retrieval filter set (top-k, embedding model, collection), and the active prompt
configuration. The scope is enforced as a hard RediSearch tag pre-filter, so the similarity
search only ever sees entries from the identical scope; similarity alone can never cross
tenants or filter sets. On top of that, a hit whose original query disagrees on a number or
reporting period ("Q3 vs Q4") is rejected, because those are different questions even at
0.95+ cosine similarity.

*Why Redis and not an in-process list:* a module-level list gave each uvicorn worker its own
cold cache. Redis shares one index across workers and survives restarts.

*Why cosine and not exact matching:* "Black-Scholes model?" and "Explain Black-Scholes" are
the same question.

The threshold is deliberately conservative — a miss costs only latency, while a wrong hit
serves a confidently incorrect answer that never touched the LLM. Every cache operation
fails open: unreachable Redis means a miss, never a request outage.

## Configuration

`src/config/settings.py` is a `pydantic-settings` model read through a cached
`get_settings()` factory. Nothing heavy is constructed at import time. Two validators exist
because their absence caused real bugs:

- **`embedding_dim` vs `embedding_model`** — a collection built with one model and queried
  with another produces meaningless similarity scores silently. A mismatch is now a hard
  failure at startup, and `verify_collection_dim` checks the live collection on
  `/health/ready`.
- **`ensemble_weights`** — must be exactly two non-negative values, because they are
  positional against `[bm25, vector]`.

Provider credentials are intentionally optional. Ingestion and retrieval do not need them;
a missing key degrades `/health/ready` rather than crash-looping the container.

## Failure and degradation

`/health/ready` classifies each dependency as required or optional. Losing an optional one
degrades quality but keeps the instance in rotation; losing a required one takes it out.

| Failure | Effect | Required? |
|---|---|---|
| Redis down | Cache and rate limiting become no-ops | optional |
| Keyword index missing | Dense-only retrieval | optional |
| Reranker unreachable / unconfigured | Fused ordering returned unranked | optional |
| Qdrant unreachable or empty | Cannot retrieve → `503` | **required** |
| Embedding model fails to load | Cannot embed → `503` | **required** |
| Generator key missing | Cannot generate → `503` | **required** |
| One PDF fails to parse | Falls back to plain-text extraction | — |

`/health/ready` never raises; it reports. Pipeline assembly is attempted inside the probe so
a construction failure is a diagnosis, not a `500`.

## Observability

- **Structured JSON logs** with a `request_id` on every record, propagated by
  `RequestContextMiddleware` and echoed in the `X-Request-ID` response header.
- **Prometheus metrics** at `/metrics` (auth-gated): HTTP latency and counts, plus
  semantic-cache hits, misses, errors, writes, and hit rate.
- Cache statistics are published through a custom collector that reads the live cache at
  scrape time, so there is one source of truth.

## Security

| Concern | Mitigation |
|---|---|
| Unauthenticated LLM spend | `X-API-Key` / JWT required when multi-tenancy is enabled; constant-time comparison |
| Abuse | Per-identity fixed-window rate limit in Redis |
| Auth bypass ordering | Auth and rate limiting share one dependency, so unauthenticated callers are rejected before consuming budget |
| Credential leakage | `.env` gitignored; API keys stored as SHA-256 hashes; internal error text never returned |
| CORS | Explicit origin allow-list; `allow_credentials=False` |
| Model/embedding drift | Dimension consistency enforced at startup and readiness |
| Container privilege | Runs as a non-root user on a slim base |

## Performance

Measured on the corpus, CPU-only, inside the container:

| Operation | Cost |
|---|---|
| PDF parse (layout-aware) | ~18 s per PDF (dominant ingestion cost) |
| Embedding (bge-small, CPU) | ~10 ms per short text |
| Hybrid recall | tens of ms |
| Cohere rerank | ~100–300 ms |
| Groq generation | ~1–3 s |
| Cache hit | ~10 ms (embed + one HNSW lookup) |

Query latency is dominated by the two external APIs — which is exactly the cost the
semantic cache exists to avoid.

## Known limitations

- **Fixed-window rate limiting** allows up to 2× the limit across a window boundary. Fine
  for cost protection, not for strict guarantees.
- **Single-writer ingestion** — concurrent ingestion against one Qdrant collection is not
  safe.
- **Embedded Qdrant mode** is single-process and file-locked; server mode is required for
  multiple workers.
- **The API must be restarted after ingestion** — retrieval components are built once per
  process, so a running container does not pick up a keyword index created after startup.
- **No query instruction** for the BGE model is applied, a measurable but unmeasured
  opportunity.
- **Evaluation depends on a hand-built golden set** — `tests/eval/run_eval.py` measures
  retrieval only; there is no automated faithfulness scoring.
- **No observability backend** — metrics are exposed but nothing scrapes them by default.
