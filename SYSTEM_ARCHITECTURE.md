# System Architecture

How this service is put together and, more importantly, why. Every section calls
out the alternative that was rejected and what it would have cost.

## 1. Components

| Component | Technology | Responsibility |
|---|---|---|
| API | FastAPI + uvicorn | HTTP surface, auth, rate limiting, validation, streaming |
| Pipeline | `src/core/rag_pipeline.py` | Retrieval → rerank → generate, cache integration |
| Vector store | Qdrant (server mode) | Child-chunk embeddings, payload-indexed by source/page |
| Keyword index | `rank_bm25` (persisted pickle) | Lexical recall over the same child chunks |
| Document store | `LocalFileStore` + pickle | Parent sections, addressed by `doc_id` |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d) | Local, CPU, no per-query API cost |
| Reranker | Cohere `rerank-v3.5` | Cross-encoder rescoring of the fused candidate set |
| Generator | Groq `openai/gpt-oss-120b` | Grounded answer generation |
| Cache | Redis Stack (HNSW vector index) | Semantic response cache + rate-limit counters |
| Frontend | React 19 + Vite, served by Caddy | Chat UI, SSE streaming, citation display |

## 2. Request lifecycle

```
RequestContextMiddleware      assign/propagate X-Request-ID, time the request
  └─ CORSMiddleware           explicit origin allow-list
      └─ authorize            verify X-API-Key, then charge the rate limit
          └─ get_pipeline     cached factory
              └─ anyio.to_thread.run_sync(pipeline.run)
                  ├─ cache.get(query)
                  ├─ hybrid_search.search(query)
                  ├─ resolve parents → dedupe → cap
                  ├─ reranker.rerank(...)
                  ├─ llm.answer(...)
                  └─ cache.set(query, answer, sources)
```

**Why the pipeline runs in a worker thread.** `pipeline.run` is synchronous and
does CPU work (embedding) plus blocking network I/O (Cohere, Groq). Calling it
directly from an `async def` handler would block the event loop for the duration of
an answer, serialising every other request.

## 3. Ingestion

### 3.1 Parsing

`DeepDocLoader` wraps `pymupdf4llm` with `page_chunks=True`, yielding one Markdown
`Document` per page with `source`, `page`, and `parser` metadata. `pymupdf4llm`
applies the same class of ONNX models RAGFlow's DeepDoc uses: document layout
recognition to classify regions, table structure recognition to convert tables to
Markdown, and reading-order reconstruction. If layout parsing fails, the loader
falls back to raw PyMuPDF text extraction rather than dropping the file.

*Rejected:* naive `page.get_text()` extraction. It destroys tables and flattens
heading structure, which are exactly the signals the chunker depends on.

### 3.2 Chunking — parent and child

Two splits happen per document:

- **Parent** — `MarkdownSectionSplitter` cuts on heading boundaries (`#`…`####`),
  with a paragraph-level fallback for sections over `PARENT_CHUNK_SIZE` or for
  documents with no heading structure at all. Parents are what the generator reads.
- **Child** — `DeepDocChildSplitter` cuts parents further into ~700-character,
  80-character-overlap windows. Children are what gets embedded and keyword-indexed.

*Why both:* a passage small enough to embed precisely is usually too small to
answer from, and a passage large enough to answer from embeds imprecisely. Small
chunks are matched; large parents are returned.

`split_documents` is used rather than `split_text` throughout, because LangChain's
`Document`-level split propagates metadata to every chunk. That is what puts
`source` and `page` on a child chunk and makes citations resolvable at all.

### 3.3 Both indexes from one split

`scripts/ingest.py` derives the BM25 corpus from the same splitters the
`ParentDocumentRetriever` uses, so a keyword hit and a dense hit refer to identical
text. If they diverged, fusion would be combining incomparable units.

### 3.4 Idempotency

Three mechanisms cooperate:

1. **Content-addressed, target-aware manifest** — keyed by corpus-relative POSIX
   path, valued with the file's SHA-256, and stamped with the collection name and
   embedding model it was built for. Unchanged files are skipped, but only while
   the target is unchanged. Skipping is a claim that vectors already exist in the
   collection being written to; if the collection name or embedding model differs,
   that claim is false and the manifest is discarded rather than trusted. A
   pre-existing manifest with no recorded target is likewise ignored — it cannot be
   verified, and re-indexing unnecessarily is far cheaper than silently producing
   an empty index. Earlier versions keyed this by absolute path, which additionally
   broke portability between a host and a container.
2. **Delete-before-add** — `delete_source(filename)` removes that file's vectors by
   payload filter before re-indexing, and its chunks are dropped from the BM25
   corpus. Without this, editing a PDF would leave the old passages retrievable.
3. **Atomic writes** — the manifest is written to a temp file and `os.replace`d, so
   an interrupted run cannot leave a half-written manifest.

Rebuilding BM25 reuses the persisted corpus rather than re-parsing the archive,
which keeps a resumed run cheap.

## 4. Retrieval

### 4.1 Hybrid recall

BM25 and dense results are fused with `EnsembleRetriever` using weighted reciprocal
rank fusion (`ENSEMBLE_WEIGHTS`, default `0.4` keyword / `0.6` dense).

*Why not dense-only:* finance text is dense with exact tokens whose meaning
embeddings blur — ticker symbols, model names (`GARCH`, `Heston`), Greek letters,
numerical parameter values. Keyword recall catches what the embedding misses, and
vice versa.

### 4.2 Relevance gating on BM25

`PersistedBM25Retriever` filters candidates by **token overlap with the query**,
not by a score threshold. `BM25Okapi` assigns negative IDFs to terms that appear in
most documents, which is routine in a small corpus or when a query is broad; a
`score > 0` rule would have silently discarded valid matches there.

### 4.3 Parent resolution

Dense retrieval returns parent sections, but BM25 indexes child chunks. Left alone,
fusion would mix units of two different sizes. Each child chunk carries the
`doc_id` of its parent, so `_resolve_parents` looks the parents up in the document
store and substitutes them, carrying the retrieval scores across so citations can
still explain why a passage was chosen.

### 4.4 Cross-document deduplication

Before reranking, passages are deduplicated on
`(source, page, sha1(content))`. Multiple children of one parent collapse to that
parent; repeated boilerplate across papers does not crowd out other results.
Without this, a single well-matched section could fill the entire context window.

The candidate set is then capped at `RERANK_MAX_CANDIDATES` before being sent to
Cohere, bounding both latency and cost.

### 4.5 Reranking

The fused top-k is rescored by Cohere `rerank-v3.5` and truncated to `top_k_rerank`.

The reranker **copies** documents rather than mutating their metadata. Retrieved
`Document` objects are shared with the retrieval layer, and in-place metadata edits
leak across requests in a long-lived process.

Reranking failures fall back to the fused ordering — an outage degrades answer
quality rather than availability.

Reranked passages below `RERANK_MIN_SCORE` are discarded before they can be cited, and
if nothing clears it the service declines without calling the generator at all. This
exists because retrieval always returns `k` results whether or not any of them are
relevant: an unanswerable query used to be answered with an honest "I don't know" *and*
a list of near-miss citations, which implies those documents support a conclusion they
do not. The reranker had already judged them irrelevant; the score was simply being
thrown away.

*Rejected:* letting the LLM's own "insufficient context" reply carry the message while
still returning every candidate as a citation. It is cheaper to implement but presents
unrelated documents as evidence, which is worse than returning no source at all.

## 5. Generation

A single `ChatPromptTemplate` pairs a finance-analyst system prompt with the
question. Context passages are labelled `[Source: <file>, page <n>]` so the model
can cite them, and the prompt instructs it to say so explicitly when the context is
insufficient rather than answering from parametric memory.

`POST /chat/stream` emits `sources` first, then `token` events, then `done`. Front-
loading citations means the UI can render provenance before the answer finishes.

## 6. Caching

`SemanticCache` embeds the incoming query and searches a Redis Stack HNSW index for
neighbours. A neighbour at or above `SEMANTIC_CACHE_THRESHOLD` (0.88) is returned
immediately, skipping retrieval, reranking, and generation.

*Why Redis and not an in-process list:* the previous implementation kept entries in
a module-level Python list. With multiple uvicorn workers each process had its own
cold cache and its own hit-rate, so the effective cache was a fraction of its
nominal size. Redis shares one index across workers and survives restarts.

*Why cosine similarity and not exact matching:* "Black-Scholes model?" and "Explain
Black-Scholes" are the same question, and exact-match caching would miss it.

Every cache operation fails open. If Redis is unreachable, `get` returns `None`,
`set` is a no-op, and the full path runs — a cache outage must never be a request
outage. The threshold is a deliberate trade-off: too low returns a wrong cached
answer, too high never fires.

## 7. Configuration

`src/config/settings.py` is a `pydantic-settings` model read through a cached
`get_settings()` factory. Nothing is instantiated at import time.

Two validators exist because their absence caused real bugs:

- **`embedding_dim` vs. `embedding_model`** — a collection built with one model and
  queried with another produces meaningless similarity scores, silently. A mismatch
  is now a hard failure at start-up, and `verify_collection_dim` checks the live
  collection against the configured dimension on `/health/ready`.
- **`ensemble_weights`** — must be exactly two non-negative values summing to a
  positive number, because they are positional against `[bm25, vector]`.

Provider credentials are intentionally **optional**. Ingestion and retrieval do not
need them. When a key is missing, the corresponding client fails to initialise and
`/health/ready` reports the service as degraded — visible in orchestration — rather
than crash-looping the container.

Secrets live in `.env` (gitignored) or the platform's secret store. Only
`.env.example` is committed.

## 8. Failure and degradation behaviour

| Failure | Behaviour |
|---|---|
| Redis down | Cache and rate limiting degrade to no-ops; requests still served |
| BM25 index missing | Dense-only retrieval; `/health/ready` reports it as an optional failure |
| Reranker unreachable or unconfigured | Fused ordering returned unranked |
| Qdrant unreachable or empty | **Required** failure → `/health/ready` returns `503` |
| Embedding model fails to load | **Required** failure → `503` |
| Groq key missing | **Required** failure → `503` |
| PDF fails to parse | Falls back to plain-text extraction; the run continues |
| Rerank API error | Logged, fused ordering used |

`/health/ready` never raises: it reports. That is why pipeline assembly is attempted
inside the probe rather than injected as a dependency — a dependency that fails
during construction would turn the readiness endpoint into a 500 instead of a
diagnosis.

## 9. Observability

- **Structured JSON logs** with a `request_id` on every record
  (`src/observability/logging.py`), propagated by `RequestContextMiddleware` and
  echoed in the `X-Request-ID` response header.
- **Prometheus metrics** at `/metrics` (auth-gated): HTTP latency and counts via
  `prometheus-fastapi-instrumentator`, plus semantic-cache hits, misses, errors,
  writes, and hit rate.
- Cache statistics are published through a custom collector that reads the live
  cache at scrape time, so there is one source of truth rather than counters
  mirrored in two places.

## 10. Security

| Concern | Mitigation |
|---|---|
| Unauthenticated LLM spend | `X-API-Key` required when `API_KEYS` is set; constant-time comparison |
| Abuse | Per-identity fixed-window rate limit in Redis, keyed on the API key with an IP fallback |
| Auth bypass ordering | Auth and rate limiting share one dependency, so unauthenticated callers are rejected before consuming budget |
| Credential leakage | `.env` gitignored; `.env.example` only; internal error text never returned to callers |
| CORS | Explicit origin allow-list; `allow_credentials=False` (wildcard origin plus credentials is unsafe and rejected by browsers) |
| Model/embedding drift | Dimension consistency enforced at start-up and at readiness |
| Container privilege | Image runs as a non-root user on a slim base |

## 11. Known limitations

- **Rate limiting is fixed-window**, so a caller can burst up to 2× the limit across
  a window boundary. Acceptable for cost protection; a sliding window or token
  bucket would be needed for stricter guarantees.
- **Single-writer ingestion.** Ingestion is not safe to run concurrently against
  the same Qdrant collection.
- **No observability backend.** Metrics are exposed but no Prometheus/Grafana stack
  is deployed. RAGFlow ships Jaeger; adding OTLP tracing would be the natural next
  step.
- **Evaluation depends on a hand-built golden set.** There is no automated
  faithfulness scoring; `tests/eval/run_eval.py` measures retrieval only.
- **Embedded Qdrant mode** is supported for convenience but is single-process and
  uses file locking; it is not suitable for a multi-worker deployment.
