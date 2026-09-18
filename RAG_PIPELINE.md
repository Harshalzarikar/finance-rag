# How the RAG Pipeline Works

An end-to-end walkthrough of what this system actually does: every stage, the real
parameters it runs with, and what happens when things fail.

Companion documents:
- [`README.md`](README.md) — setup, configuration, and how to run it
- [`SYSTEM_ARCHITECTURE.md`](SYSTEM_ARCHITECTURE.md) — why each design choice was made, and what was rejected

---

## 1. What the system does

It answers natural-language questions about a corpus of quantitative finance research
papers (arXiv q-fin), grounded strictly in retrieved passages, with citations back to
the source document and page.

There are two pipelines that never run at the same time:

- **Ingestion** — offline, batch, writes the indexes. Run by `scripts/ingest.py`.
- **Query** — online, per-request, reads the indexes. Served by the FastAPI app.

Both must agree on one thing: **how text becomes chunks**. If ingestion and query
disagreed about chunking or embeddings, retrieval would compare incomparable things.
That is why both share the same splitter and embedding factories in `src/`.

---

## 2. Component map

```
                    ┌────────────────────────────────────────────┐
   PDF corpus  ───► │  scripts/ingest.py                         │
   (real_pdfs/)     │   DeepDocLoader → chunkers → embeddings      │
                    │   → Qdrant, docstore, BM25, manifest         │
                    └───────────────┬────────────────────────────┘
                                    │ writes
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                      ▼
        ┌──────────┐        ┌───────────────┐      ┌─────────────┐
        │  Qdrant  │        │ docstore      │      │ BM25 pickle │
        │ (child   │        │ (PickleFile   │      │ (child      │
        │ vectors) │        │  Store:       │      │  chunks +   │
        │          │        │  parents)     │      │  metadata)  │
        └────┬─────┘        └───────┬───────┘      └──────┬──────┘
             │                      │                     │
             └──────────┬───────────┴─────────────────────┘
                        │ reads
                ┌───────▼──────────────────────────────┐
                │  FastAPI  (src/api/)                 │
                │   auth → rate limit → pipeline       │
                │   src/core/rag_pipeline.py           │
                └───────┬──────────────────────────────┘
                        │
                  ┌─────▼─────┐
                  │  Redis    │ semantic cache
                  │  Stack    │ + rate-limit counters
                  └───────────┘
```

| Module | Responsibility |
|---|---|
| `src/ingestion/deepdoc_loader.py` | PDF → Markdown per page, layout-aware |
| `src/ingestion/chunker.py` | Parent (section) and child (passage) splitting |
| `src/retrieval/vector_store.py` | Embeddings, Qdrant client, collection setup, docstore |
| `src/retrieval/bm25.py` | Keyword index: tokenizer, build, persist, retrieve |
| `src/retrieval/hybrid_search.py` | Fuses keyword + dense results |
| `src/llm/reranker.py` | Cohere cross-encoder reranking |
| `src/llm/generator.py` | Groq chat model |
| `src/cache/semantic_cache.py` | Redis vector cache for whole answers |
| `src/core/rag_pipeline.py` | Orchestrates the query path |
| `src/api/` | HTTP surface, auth, rate limits, health, metrics |

---

## 3. Ingestion pipeline

Entry point: `scripts/ingest.py`

```bash
python scripts/ingest.py                        # every PDF in RAW_PDFS_DIR
python scripts/ingest.py --pdf real_pdfs/x.pdf  # one file
python scripts/ingest.py --reset --limit 5      # wipe everything, then ingest 5
python scripts/ingest.py --offset 100 --limit 50
python scripts/ingest.py --concurrency 4        # parallel parsing
python scripts/ingest.py --verify               # drift check, no writes
```

### Stage 1 — Parse (PDF → Markdown per page)

`DeepDocLoader` wraps `pymupdf4llm.to_markdown(path, page_chunks=True)`, which runs
ONNX layout-recognition models:

- **Document layout recognition** classifies page regions (header, paragraph, table, figure, caption, footer)
- **Table structure recognition** converts tables into Markdown tables rather than raw text
- **Reading-order detection** assembles blocks in natural reading order

Each page becomes a separate LangChain `Document` with metadata:

```python
{
    "source": "0704.2244v13.pdf",  # basename — this is what citations show
    "file_path": "/pdfs/0704.2244v13.pdf",
    "page": 4,  # from pymupdf4llm's "page_number"
    "parser": "deepdoc_layout",
}
```

Blank pages are skipped. If layout parsing throws for any reason, the loader falls
back to raw PyMuPDF text extraction and marks `"parser": "pymupdf_fallback"` — a bad
PDF degrades to worse text rather than being dropped.

> Note: the page number comes from `page_number` in `pymupdf4llm`'s metadata. Reading
> `page` instead yields `None` on every chunk, which silently empties page numbers from
> all citations. This is covered by `tests/test_deepdoc_loader.py`.

### Stage 2 — Chunk (two-level split)

Two splitters run in sequence, both configured from `Settings`:

**Parent split — `MarkdownSectionSplitter`** (`PARENT_CHUNK_SIZE=4000`)

Splits on Markdown headings matching `^#{1,4}\s+.+`. Each heading starts a new section.
A section longer than 4000 characters is further split on blank lines (paragraph
boundaries). A document with no headings at all falls back to paragraph splitting
entirely.

**Child split — `DeepDocChildSplitter`** (`CHILD_CHUNK_SIZE=700`, `CHILD_CHUNK_OVERLAP=80`)

A `RecursiveCharacterTextSplitter` over parents, trying separators in order:
`"\n\n"` → `"\n"` → `". "` → `"! "` → `"? "` → `" "` → `""`. So it prefers to break at
paragraph, then sentence, then word.

**Why both sizes exist:** a passage small enough to embed precisely (~700 chars) is
usually too small to answer from, and a passage large enough to answer from embeds
imprecisely. Small chunks are matched; large parents are returned.

Critically, `split_documents` is used rather than `split_text`, because LangChain's
document-level split propagates metadata to every resulting chunk. That is what puts
`source` and `page` onto a child chunk — without it, citations would be unresolvable.

### Stage 3 — Index (three stores, one split)

```
parents  ──► PickleFileStore (docstore), keyed by a generated doc_id
children ──► embeddings ──► Qdrant
children ──► BM25 pickle (text + metadata)
```

`ParentDocumentRetriever.add_documents()` performs the parent/child split internally,
and then the ingestion script derives the BM25 corpus from **the same splitters**
(`retriever.parent_splitter`, `retriever.child_splitter`), so a keyword hit and a dense
hit refer to identical text.

Each child chunk carries `doc_id` — the key of its parent in the docstore. This is the
link that lets retrieval expand a small keyword hit back into a full section.

### Stage 4 — Idempotency

Four mechanisms:

1. **Content-addressed manifest** — keyed by path relative to the corpus root (POSIX
   separators), valued with the file's SHA-256. An unchanged file is skipped.
2. **Target stamping** — the manifest records the collection name and embedding model
   it was built for. If either differs, the whole manifest is discarded and everything
   re-indexes. Without this, switching models would silently skip every file while its
   vectors lived in a different collection.
3. **Delete-before-add** — re-indexing a file first deletes its existing vectors by
   payload filter (`metadata.source`), and its old chunks are dropped from the BM25
   corpus. Otherwise editing a PDF would leave stale passages retrievable.
4. **Atomic writes** — the manifest is written to a temp file and `os.replace`d, so an
   interrupted run cannot leave it half-written. Progress is saved after every file, so
   a killed run resumes.

> Limitation: the manifest tracks file contents and the target, **not the parsing or
> chunking code**. After changing anything in `src/ingestion/`, re-run with `--reset`,
> or unchanged files keep their old chunks.

### Stage 5 — Verify

`--verify` compares the manifest against reality and exits non-zero on drift:

```
Manifest files  : 3
Expected chunks : 593
Qdrant points   : 593
BM25 chunks     : 593
No drift detected.
```

### Measured behaviour

A 3-PDF smoke run on the real corpus:

```
Found 3 PDF(s) to ingest (concurrency=2).
Indexed 3 PDF(s), skipped 0 unchanged, 0 failed (75 pages, 53.6s).
BM25 index: 0 retained + 593 new chunks.
```

≈ 18 s/PDF single-threaded on CPU, ≈ 7.9 child chunks per page. The full corpus is
1,191 PDFs, so expect several hours — use `--concurrency` and run it detached.

---

## 4. Query pipeline

Entry point: `POST /chat` or `POST /chat/stream` → `src/core/rag_pipeline.py`

```
query
  │
  ├─► 0. SEMANTIC CACHE ────────────── hit ──► return cached answer
  │        (Redis HNSW, cosine ≥ 0.88)
  │   miss
  │
  ├─► 1. HYBRID RECALL
  │        ├── BM25 keyword      k = 8
  │        └── Qdrant dense      fetch_k = 20
  │        └── weighted RRF fusion (0.4 keyword / 0.6 dense)
  │
  ├─► 2. PARENT RESOLUTION
  │        child chunks → their parent sections (docstore lookup)
  │
  ├─► 3. DEDUPLICATION
  │        collapse on (source, page, sha1(content))
  │        cap at RERANK_MAX_CANDIDATES = 40
  │
  ├─► 4. CROSS-ENCODER RERANK
  │        Cohere rerank-v3.5 → top_n (default 5)
  │
  ├─► 4b. RELEVANCE FLOOR
  │        drop passages scoring < RERANK_MIN_SCORE
  │        if none survive → decline, citing nothing, skip the LLM
  │
  └─► 5. GENERATION
           Groq openai/gpt-oss-120b, temperature 0.0
           context labelled [Source: <file>, page <n>]
           └──► answer + citations → written back to cache
```

### Step 0 — Semantic cache

The incoming query is embedded with **the same model used for retrieval** and compared
against previously answered queries in a Redis Stack HNSW index.

A neighbour at or above `SEMANTIC_CACHE_THRESHOLD` (0.88) cosine similarity returns its
stored answer immediately — skipping retrieval, reranking, and generation entirely.
Otherwise the full path runs and the result is written back with a TTL
(`SEMANTIC_CACHE_TTL_SECONDS`, default 24 hours).

On a hit the response carries `"cached": true` and `cache_similarity`.

Why a threshold of 0.88: BGE-family models produce a compressed similarity
distribution (roughly `[0.6, 1.0]`), so a score above 0.5 does **not** mean two texts
are similar. BAAI's own guidance is to pick a threshold from `0.8 / 0.85 / 0.9` based on
your data. 0.88 sits inside that band.

Every cache operation **fails open**: an unreachable Redis returns a miss and the full
path runs. A cache outage must never become a request outage.

### Step 1 — Hybrid recall

Two retrievers run and are fused with weighted reciprocal rank fusion:

| Retriever | What it searches | Recall |
|---|---|---|
| BM25 (`rank_bm25.BM25Okapi`) | child chunks, lexical | `BM25_K` = 8 |
| Qdrant (dense) | child chunks, semantic | `VECTOR_FETCH_K` = 20 |

Then `EnsembleRetriever` fuses them with `ENSEMBLE_WEIGHTS` = `0.4,0.6`
(keyword / dense).

**Why not dense-only:** finance text is dense with exact tokens whose meaning
embeddings blur — ticker symbols, model names (`GARCH`, `Heston`), Greek letters,
numeric parameter values. Keyword recall catches what the embedding misses, and vice
versa.

**BM25 tokenization.** One canonical tokenizer, `[a-z0-9]+` over lowercased text, is
shared by index construction and querying, so the two can never drift. It splits
`"Black-Scholes GARCH(1,1)"` into `["black","scholes","garch","1","1"]` — meaning
"Black-Scholes" and "black scholes" match each other.

**BM25 relevance gating.** A candidate is kept only if it shares at least one token with
the query. It deliberately does **not** filter on a score threshold: `BM25Okapi` assigns
negative IDFs to terms appearing in most documents, which is routine in a small corpus
or for a broad query, and a `score > 0` rule would silently discard valid matches there.

### Step 2 — Parent resolution

Dense retrieval already returns parent sections, but BM25 indexes child chunks — so
without this step fusion would mix units of two very different sizes.

Every child carries `doc_id`. `_resolve_parents` batch-loads the parents from the
docstore and substitutes them, carrying the retrieval scores across so citations can
still explain *why* a passage was selected:

```python
metadata = {**parent.metadata, **document.metadata}
```

Documents that already are parents (no `doc_id`) pass through untouched.

### Step 3 — Deduplication and capping

Passages are deduplicated on `(source, page, sha1(content))`, then truncated to
`RERANK_MAX_CANDIDATES` (40).

Two effects: multiple children of one parent collapse into that parent, and repeated
boilerplate across papers does not crowd out other results. Without this, one
well-matched section could fill the entire context window. The cap also bounds reranker
latency and cost.

### Step 4 — Cross-encoder reranking

The fused candidate set is rescored by Cohere `rerank-v3.5` (configurable via
`RERANK_MODEL`) and truncated to `top_k_rerank` (request parameter, default 5, validated
to 1–20).

A bi-encoder compares embeddings independently; a cross-encoder reads query and passage
**together**, which is far more accurate and far too slow to run over the whole corpus.
That is exactly why it runs last, on a small candidate set.

Two implementation details:

- Reranked documents are **copies**, not the originals. Retrieved `Document` objects are
  shared with the retrieval layer, and in-place metadata edits leak across requests in a
  long-lived process.
- Any failure (API error, missing key, disabled by config) falls back to the fused
  ordering. Reranker trouble degrades answer quality, it does not cause an outage.

### Step 4b — Relevance floor

Passages the reranker scored below `RERANK_MIN_SCORE` (default `0.30`) are dropped
before they can become citations.

Without this, an unanswerable query still returned a full list of citations — the
least-irrelevant passages in the corpus — which reads as evidence for an answer that
does not exist. The reranker already knew they were irrelevant; the score was simply
being ignored. The separation is wide and unambiguous:

| Query | Top rerank scores |
|---|---|
| answerable ("minimum probability of lifetime ruin") | 0.92, 0.90, 0.89, 0.88, 0.87 |
| not in corpus ("heston model") | 0.07, 0.05, 0.04 |

If **nothing** clears the floor, the pipeline returns a fixed decline message with
`sources: []` and **does not call the LLM at all** — there is nothing to ground an
answer in, and the call would only produce a paraphrase of "I don't know" at the cost
of a request. That response is also deliberately **not cached**, so it does not outlive
a re-index that makes the question answerable.

If no document carries a reranker score at all — reranking disabled, key missing, or
the API failed — the floor cannot be applied and every passage is kept, preserving the
previous behaviour.

### Step 5 — Generation

Context is assembled with explicit labels:

```
[Source: 0704.2244v13.pdf, page 4]
<passage text>

---

[Source: 0705.3760v1.pdf, page 11]
<passage text>
```

The prompt (`SYSTEM_PROMPT` in `rag_pipeline.py`) instructs the model to answer only
from that context, to cite the `[Source: ...]` labels, and to say so explicitly when the
context is insufficient rather than answering from memory.

Generation runs at `temperature=0.0` for reproducibility.

Real response from the 3-PDF corpus:

```json
{
  "answer": "The minimum probability of lifetime ruin is the smallest possible chance ...",
  "sources": [
    {"source": "0704.2244v13.pdf", "page": 4,  "score": 0.9176, "snippet": "..."},
    {"source": "0704.2244v13.pdf", "page": 4,  "score": 0.8972, "snippet": "..."},
    {"source": "0705.3760v1.pdf",  "page": 11, "score": 0.8655, "snippet": "..."}
  ],
  "cached": false,
  "request_id": "24c8801eee2d477a"
}
```

### Streaming variant

`POST /chat/stream` emits Server-Sent Events:

```
data: {"type": "sources", "sources": [...], "cached": false}
data: {"type": "token", "value": "The "}
data: {"type": "token", "value": "minimum "}
data: {"type": "done", "cached": false}
```

Citations are emitted **before** the first token so the UI can render provenance
immediately. A cache hit is replayed as a single token. Errors after the headers are
sent are reported in-band as `{"type": "error", ...}`.

---

## 5. Embeddings

| Property | Value |
|---|---|
| Model | `BAAI/bge-small-en-v1.5` |
| Dimension | **384** |
| Architecture | BERT, 12 layers, 12 heads, 384 hidden |
| Max input | 512 tokens |
| Pooling | CLS token |
| Normalisation | built into the model, and applied again via `normalize_embeddings=True` |
| Licence | MIT |
| MTEB retrieval score | 51.68 (vs ada-002's 49.25) |

Chosen because it runs on CPU with no per-query API cost and no network dependency,
while still beating OpenAI's `text-embedding-ada-002` on retrieval benchmarks. It is
baked into the Docker image at build time.

Distance metric is **cosine**; because vectors are L2-normalised, cosine reduces to a
dot product.

`src/config/settings.py` keeps an `EMBEDDING_DIMS` registry and **refuses to start** if
`EMBEDDING_DIM` disagrees with `EMBEDDING_MODEL`. `verify_collection_dim()` performs the
same check against the live collection at readiness time. Silent dimension drift
produces meaningless similarity scores rather than errors — hence the hard guard.

> Current gap: BGE models document a query instruction
> (`Represent this sentence for searching relevant passages: `) for short-query →
> long-passage retrieval, which is exactly this workload. It is **not** applied: only
> `encode_kwargs` is set, so queries and documents are encoded identically. v1.5 was
> tuned so omitting it causes only slight degradation, so this is a marginal
> opportunity rather than a defect — and it should be measured with
> `tests/eval/run_eval.py` before changing, because it alters cached query embeddings.

---

## 6. Storage layout

| Store | Contents | Keyed by |
|---|---|---|
| **Qdrant collection** | child-chunk vectors + payload (`page_content`, `metadata`) | generated point IDs |
| **Docstore** (`PickleFileStore` on `LocalFileStore`) | parent sections, pickled | `doc_id` |
| **BM25 pickle** | child chunk texts + metadata + the fitted index | — |
| **Manifest** | file hashes, target, per-file chunk counts | corpus-relative path |
| **Redis** | cached answers (HNSW index) + rate-limit counters | query hash |

Qdrant payload indexes exist on `metadata.source` (keyword) and `metadata.page`
(integer) so delete-by-source and filtering are fast.

In Docker, `doc_store`, `bm25_index.pkl`, and the manifest all live on the `rag_data`
volume at `/data`. That directory is created and chowned to the unprivileged container
user in the image, because Docker seeds a fresh named volume from the image's
directory — including ownership. Without it the volume is root-owned and the app cannot
write.

---

## 7. HTTP API

| Endpoint | Auth | Description |
|---|---|---|
| `POST /chat` | required | Answer a question with citations |
| `POST /chat/stream` | required | Same, as Server-Sent Events |
| `GET /health/live` | public | Process is up |
| `GET /health/ready` | public | Probes every dependency; `503` if a required one is down |
| `GET /metrics` | required | Prometheus exposition |
| `GET /` | public | Service descriptor |

**Request body**

```json
{ "query": "What is a volatility smile?", "top_k_rerank": 5 }
```

`query` is trimmed, must be non-empty, and is capped at `MAX_QUERY_CHARS` (2000).
`top_k_rerank` is validated to 1–20.

**Auth.** `X-API-Key` compared with `secrets.compare_digest` against `API_KEYS`
(comma-separated). **An empty `API_KEYS` disables authentication entirely** — intended
for local development only.

**Rate limiting.** Fixed one-minute window in Redis, keyed on the API key with a client
IP fallback. Auth and rate limiting share a single dependency, so an unauthenticated
caller is rejected *before* consuming budget. Fails open if Redis is unreachable.

**Error codes.** `401` auth, `422` validation, `429` rate limit (with `Retry-After`),
`500` internal. Internal error text is never returned to the caller — it goes to the
logs, correlated by the `X-Request-ID` response header.

---

## 8. Failure and degradation

`/health/ready` classifies every dependency as **required** or **optional**. Losing an
optional one degrades quality but keeps the instance in rotation; losing a required one
takes it out.

| Failure | Effect | Required? |
|---|---|---|
| Redis down | Cache and rate limiting become no-ops; every request runs the full path | optional |
| BM25 index missing | Dense-only retrieval — no keyword recall | optional |
| Reranker unreachable / unconfigured | Fused ordering returned unranked | optional |
| Qdrant unreachable or collection missing | Cannot retrieve → `503` | **required** |
| Embedding model fails to load | Cannot embed → `503` | **required** |
| Groq key missing | Cannot generate → `503` | **required** |
| One PDF fails to parse | Falls back to plain-text extraction; run continues | — |

A real readiness response, with the keyword index not yet loaded:

```json
{
  "status": "ok",
  "dependencies": [
    {"name": "qdrant",        "ok": true,  "required": true,  "detail": "collection 'finance_documents_v1' dim=384"},
    {"name": "embeddings",    "ok": true,  "required": true,  "detail": "BAAI/bge-small-en-v1.5 (384d)"},
    {"name": "groq",          "ok": true,  "required": true,  "detail": "client configured for openai/gpt-oss-120b"},
    {"name": "bm25",          "ok": false, "required": false, "detail": "index missing or unloaded — retrieval is dense-only"},
    {"name": "cohere_rerank", "ok": true,  "required": false, "detail": "model rerank-v3.5"},
    {"name": "redis",         "ok": true,  "required": false, "detail": "reachable"}
  ]
}
```

`/health/ready` never raises. Pipeline assembly is attempted *inside* the probe, so a
construction failure is reported as a diagnosis rather than turning the endpoint into a
`500`.

> **The API must be restarted after ingestion.** Retrieval components are built once per
> process and cached, so a running `api` container will not pick up a BM25 index created
> after it started — it will keep reporting `bm25` as a non-required failure and run
> dense-only. `docker compose restart api` after every ingest fixes it.

---

## 9. Configuration reference

Every value below is settable via environment or `.env` (environment wins).

**Credentials** — optional at start-up; a missing key degrades readiness rather than
crashing the process.

| Variable | Default |
|---|---|
| `GROQ_API_KEY` | *(empty)* |
| `GROQ_MODEL_NAME` | `openai/gpt-oss-120b` |
| `COHERE_API_KEY` | *(empty)* |
| `RERANK_MODEL` | `rerank-v3.5` |
| `RERANK_ENABLED` | `true` |
| `RERANK_MAX_CANDIDATES` | `40` |
| `RERANK_MIN_SCORE` | `0.30` — below this a passage is not cited; if none clear it, the service declines rather than citing unrelated documents |

**Security**

| Variable | Default | Note |
|---|---|---|
| `API_KEYS` | *(empty)* | **Empty disables auth** |
| `CORS_ORIGINS` | `http://localhost:5173` | comma-separated |
| `RATE_LIMIT_PER_MINUTE` | `30` | |
| `MAX_QUERY_CHARS` | `2000` | |

**Storage and retrieval**

| Variable | Default |
|---|---|
| `QDRANT_URL` | `http://localhost:6333` (empty ⇒ embedded, single-process) |
| `QDRANT_COLLECTION_NAME` | `finance_documents_v1` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` |
| `EMBEDDING_DIM` | `384` (validated against the model) |
| `PARENT_CHUNK_SIZE` | `4000` |
| `CHILD_CHUNK_SIZE` | `700` |
| `CHILD_CHUNK_OVERLAP` | `80` |
| `BM25_K` | `8` |
| `VECTOR_FETCH_K` | `20` |
| `ENSEMBLE_WEIGHTS` | `0.4,0.6` (`[bm25, vector]`, exactly two values) |
| `DOC_STORE_DIR` | `./doc_store_local` |
| `RAW_PDFS_DIR` | `./real_pdfs` |
| `BM25_INDEX_FILE` | `bm25_index.pkl` |
| `INGESTION_MANIFEST_FILE` | `ingestion_manifest.json` |

**Cache and observability**

| Variable | Default |
|---|---|
| `SEMANTIC_CACHE_ENABLED` | `true` |
| `SEMANTIC_CACHE_THRESHOLD` | `0.88` |
| `SEMANTIC_CACHE_TTL_SECONDS` | `86400` |
| `SEMANTIC_CACHE_MAX_CANDIDATES` | `3` |
| `REDIS_URL` | `redis://localhost:6379/0` (must be Redis **Stack**) |
| `LOG_LEVEL` | `INFO` |
| `LOG_JSON` | `true` |

---

## 10. Performance characteristics

Measured on the real corpus, CPU-only, inside the container:

| Operation | Cost |
|---|---|
| PDF parse (layout-aware) | ~18 s per PDF (dominant cost) |
| Child chunks per page | ~7.9 |
| Embedding (bge-small, CPU) | ~10 ms per short text |
| Query: hybrid recall | tens of ms |
| Query: Cohere rerank | network-bound, ~100–300 ms |
| Query: Groq generation | network-bound, ~1–3 s |
| Query: cache hit | ~10 ms (embed + one HNSW lookup) |

Parsing dominates ingestion, which is why `--concurrency` parallelises parsing while
indexing stays serial (the embedding model and vector store are not thread-safe).
Query latency is dominated by the two external APIs — which is precisely the cost the
semantic cache exists to avoid.

---

## 11. Known limitations

- **Not yet fully ingested.** The pipeline is verified on 3 of 1,191 PDFs.
- **No query instruction** for the BGE model (see §5) — a measurable but unmeasured
  opportunity.
- **Fixed-window rate limiting** allows up to 2× the limit across a window boundary. Fine
  for cost protection, not for strict guarantees.
- **Single-writer ingestion.** Concurrent ingestion runs against one collection are not
  safe.
- **Embedded Qdrant mode** is single-process and file-locked; server mode is required for
  multiple workers.
- **Manifest does not track code version**, so changing parsing or chunking requires
  `--reset`.
- **Evaluation depends on a hand-built golden set.** `tests/eval/run_eval.py` measures
  retrieval (recall@k, MRR, rerank lift) only; there is no automated faithfulness scoring.
- **No observability backend.** Metrics are exposed at `/metrics` but nothing scrapes
  them.

---

## 12. Quick reference

```bash
# Ingest
python scripts/ingest.py --reset --limit 5     # smoke test
python scripts/ingest.py --concurrency 4       # full corpus
python scripts/ingest.py --verify              # drift check

# Ask
curl -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
     -d '{"query":"What is a volatility smile?"}' localhost:8000/chat

curl -N -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
     -d '{"query":"Explain GARCH"}' localhost:8000/chat/stream

# Check
curl -fsS localhost:8000/health/ready | jq
curl -fsS -H "X-API-Key: $KEY" localhost:8000/metrics | head

# Evaluate retrieval
python tests/eval/run_eval.py --k 5
```

## 13. Glossary

| Term | Meaning here |
|---|---|
| **Parent** | A document section (up to 4000 chars), split on Markdown headings. What the LLM reads. |
| **Child** | A ~700-char passage from within a parent. What gets embedded and keyword-indexed. |
| **`doc_id`** | The docstore key linking a child to its parent. |
| **Recall** | The retrieval stage that finds candidates. |
| **Rerank** | The cross-encoder stage that reorders candidates. |
| **Fusion** | Combining keyword and dense result lists (weighted reciprocal rank fusion). |
| **Fail open** | On error, continue with reduced functionality rather than failing the request. |
| **Target** | The `(collection, embedding model)` pair a manifest was built against. |
