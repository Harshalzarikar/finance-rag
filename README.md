# Quantitative Finance RAG

A production-oriented retrieval-augmented generation service over a corpus of
quantitative finance research papers. Layout-aware parsing, parent/child chunking,
hybrid keyword + dense retrieval, cross-encoder reranking, grounded citations, and
a Redis-backed semantic cache.

Built on design ideas borrowed from [RAGFlow](https://github.com/infiniflow/ragflow)
rather than on RAGFlow itself — see [Attribution](#attribution).

## Architecture

```
                 browser
                    │  HTTPS
                    ▼
        ┌───────────────────────────┐
        │  frontend (Caddy)         │   serves the SPA
        │  /api/*  ──► api:8000     │   reverse-proxies the API
        └───────────┬───────────────┘
                    ▼
        ┌───────────────────────────┐
        │  api (FastAPI)            │
        │   auth · rate limit       │
        │   semantic cache          │
        │   hybrid recall           │
        │   rerank · generate       │
        └──┬──────────┬─────────┬───┘
           ▼          ▼         ▼
       qdrant      redis    Groq / Cohere
     (vectors)   (cache,    (generation,
                  limits)    reranking)
```

Query path:

```
query
  └─► semantic cache ──hit──► return cached answer
        │ miss
        ├─► BM25 keyword recall  (k=8)
        ├─► Qdrant dense recall  (k=20 candidates)
        ├─► fuse with weighted reciprocal rank fusion (0.4 / 0.6)
        ├─► expand child chunks back to their parent sections
        ├─► drop duplicate passages
        ├─► Cohere cross-encoder rerank (top_n)
        └─► Groq generation with [Source: …] labels ──► answer + citations
```

Ingestion path:

```
PDF ─► DeepDoc-style layout parsing (pymupdf4llm, ONNX DLR/TSR) ─► Markdown per page
    ─► parent split on Markdown heading boundaries
    ─► child split (~700 chars, 80 overlap)
    ├─► embeddings ─► Qdrant
    ├─► parents ─────► local document store
    └─► children ────► BM25 keyword index
```

Both indexes are built from a single split, so a keyword hit and a dense hit refer
to the same unit of text and carry the same citation metadata.

See [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) for the reasoning behind each
of these choices.

## Requirements

- Python 3.13
- Docker + Docker Compose (for the containerised stack); a local run needs Qdrant
  and Redis Stack reachable, or you can run Qdrant in embedded mode
- API keys for Groq (generation) and Cohere (reranking). Both are optional at
  start-up: without them the service reports `degraded` on `/health/ready`
  instead of failing to boot.

## Configuration

Every setting has a default and is documented in [`.env.example`](.env.example).
Copy it and fill in the values:

```bash
cp .env.example .env
```

Environment variables take precedence over `.env`, which is useful for overriding
a single value in a container or a test.

The settings that most often need changing:

| Variable | Default | Notes |
|---|---|---|
| `API_KEYS` | *(empty)* | Comma-separated. **Empty disables authentication** — set this in any deployment reachable by others. |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated browser origins. |
| `QDRANT_URL` | `http://localhost:6333` | Leave empty to use embedded on-disk Qdrant (single process only). |
| `REDIS_URL` | `redis://localhost:6379/0` | Must be a **Redis Stack** server; the plain `redis` image has no vector index. |
| `GROQ_API_KEY` / `COHERE_API_KEY` | *(empty)* | Required for generation and reranking respectively. |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `BAAI/bge-small-en-v1.5` / `384` | Validated against each other at start-up; a mismatch is a hard error. |
| `SEMANTIC_CACHE_THRESHOLD` | `0.88` | Cosine similarity required for a cache hit. |

## Running locally

```bash
python -m venv venv
venv/Scripts/activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements-dev.txt

# Infrastructure (or run the full stack with docker compose instead)
docker compose up -d qdrant redis

# Build the indexes — the first thing to do on a fresh checkout
python scripts/ingest.py --limit 5     # smoke test
python scripts/ingest.py               # full corpus

# Serve
uvicorn src.api.main:app --reload --port 8000
```

The API is then at `http://localhost:8000` with interactive docs at `/docs`.

```bash
curl -H "X-API-Key: $API_KEYS" -H 'Content-Type: application/json' \
     -d '{"query": "What is a volatility smile?"}' \
     http://localhost:8000/chat

# Server-Sent Events, citations first then tokens
curl -N -H "X-API-Key: $API_KEYS" -H 'Content-Type: application/json' \
     -d '{"query": "Explain GARCH"}' \
     http://localhost:8000/chat/stream
```

### Frontend development

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` to `http://localhost:8000` (override with `VITE_PROXY_TARGET`),
so the browser always talks to a single relative base path — the same one used in
production.

## Ingestion

```bash
python scripts/ingest.py                        # everything in RAW_PDFS_DIR
python scripts/ingest.py --pdf real_pdfs/x.pdf  # a single file
python scripts/ingest.py --reset --limit 5      # wipe, then ingest the first 5
python scripts/ingest.py --offset 100 --limit 50
python scripts/ingest.py --concurrency 4        # parallel PDF parsing
python scripts/ingest.py --verify               # manifest vs. stores drift check
```

Ingestion is idempotent and resumable:

- Files are keyed in the manifest by **path relative to the corpus root** plus a
  SHA-256 of their contents, so an unchanged file is skipped and the manifest
  stays portable between a host and a container.
- The manifest records **which collection and embedding model** it was built
  against. Pointing at a different collection or model discards it and re-indexes
  everything — otherwise the run would skip files whose vectors do not exist.
- Re-ingesting a changed file first deletes that file's existing vectors and
  keyword chunks, so nothing is duplicated.
- Progress is written atomically after every file; an interrupted run resumes
  where it stopped.

The manifest tracks file contents and the target collection, **not the parsing or
chunking code**. After changing `src/ingestion/`, re-run with `--reset` so every
document is re-chunked; otherwise unchanged files are skipped and keep their old
chunks.

PDF parsing is the bottleneck and is parallelised with `--concurrency`; indexing
itself stays serial because the embedding model and vector store are not
thread-safe.

## API

| Endpoint | Auth | Description |
|---|---|---|
| `POST /chat` | required | Answer a question, returning citations. |
| `POST /chat/stream` | required | Same, streamed as Server-Sent Events. |
| `GET /health/live` | public | Process liveness. |
| `GET /health/ready` | public | Probes Qdrant, embeddings, the LLM client, the keyword index, the reranker, and Redis. Returns `503` when a **required** dependency is down. |
| `GET /metrics` | required | Prometheus exposition, including semantic-cache statistics. |
| `GET /` | public | Service descriptor. |

`/health/ready` distinguishes required dependencies (Qdrant, embeddings, the
generator) from optional ones (keyword index, reranker, cache). Losing an optional
dependency degrades quality but keeps the service in rotation; losing a required
one takes it out.

Errors are returned as `{"detail": "..."}` with `401` (auth), `422` (validation),
`429` (rate limit, with `Retry-After`), or `500`. Internal error text is never
leaked — it goes to the logs, correlated by the `X-Request-ID` response header.

## Deployment

```bash
cp .env.example .env      # then edit: set API_KEYS, provider keys, CORS_ORIGINS
docker compose up -d --build
docker compose --profile tools run --rm ingest        # build the indexes
docker compose restart api                            # load the new keyword index
curl -fsS localhost:8000/health/ready
```

The stack runs `api`, `qdrant`, `redis`, `frontend` (Caddy), and a one-shot
`ingest` job behind the `tools` profile.

> **Restart the API after ingesting.** The retrieval stack is built once per
> process and cached, so a running `api` container will not pick up a BM25 index
> created after it started — `/health/ready` reports `bm25` as a non-required
> failure and retrieval silently runs dense-only. `docker compose restart api`
> after every ingestion run fixes it.

Additional notes for a real deployment:

- **TLS.** The frontend container terminates plain HTTP on port 80. Put it behind
  a TLS terminator (Caddy automatic HTTPS, nginx, or a cloud load balancer), or
  add a domain to `frontend/Caddyfile` and let Caddy issue a certificate.
- **Stop publishing the API port.** `API_PORT=8000` is published for debugging.
  Remove that mapping to force all traffic through the proxy.
- **Secrets.** `.env` is gitignored; only `.env.example` is committed. Use your
  platform's secret store (Docker secrets, Vault, SOPS) to inject values in
  production rather than shipping a `.env` file.
- **Back up the volumes.** `qdrant_data` (vectors), `redis_data` (cache — safe to
  lose), and `rag_data` (parent document store + manifest + BM25 index).
- **Model warmth.** The embedding model is baked into the image at build time, so
  containers do not need network access to Hugging Face at runtime.

## Quality gates

```bash
ruff check .            # lint
ruff format --check .   # formatting
mypy                    # types
pytest -q               # tests
```

`make check` runs all four. The test suite needs no network: every external client
is stubbed.

### Evaluating retrieval

```bash
cp tests/eval/golden_queries.example.json tests/eval/golden_queries.json
# fill in verified relevant_sources, then:
python tests/eval/run_eval.py --k 5
```

Reports `recall@k`, `MRR`, and the MRR lift from reranking. It runs offline against
the live index, so it is deterministic and cheap enough to run after each
ingestion. Queries whose ground truth you have not verified are reported as
skipped rather than counted.

## Project layout

```
src/
  api/            FastAPI app, routes, schemas, auth/rate-limit deps, health probing
  cache/          Redis semantic cache
  config/         validated settings
  core/           pipeline orchestration
  ingestion/      layout-aware loader and parent/child chunkers
  llm/            generation and reranking
  observability/  structured logging, request IDs, Prometheus metrics
  retrieval/      Qdrant vector store, BM25 index, hybrid fusion
scripts/ingest.py CLI ingestion pipeline
frontend/         React + Vite SPA served by Caddy
tests/            unit and API tests
tests/eval/       retrieval evaluation harness
reference/ragflow upstream RAGFlow assets kept for reference only
```

## Attribution

The retrieval design follows patterns established by
[RAGFlow](https://github.com/infiniflow/ragflow) (Apache-2.0):

- **Layout-aware parsing** — `pymupdf4llm` runs the same class of ONNX document
  layout recognition and table structure recognition models RAGFlow's DeepDoc uses.
- **Parent/child chunking** — large sections are kept for generation context while
  small child chunks are embedded and matched, so retrieval stays precise without
  starving the model of context.
- **Multiple recall** — keyword and dense candidates are fused rather than picking
  one, which matters for finance text where exact terms (ticker symbols, model
  names, Greek letters) carry meaning that embeddings blur.
- **Fused reranking** — a cross-encoder rescores the fused set before generation.

RAGFlow itself is not deployed. Its compose file and environment template are kept
under `reference/ragflow/` purely as reference and are not part of this stack.

## License

See the repository for licensing details.
