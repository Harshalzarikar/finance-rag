# Quantitative Finance RAG

**New to this repo?** Read [START_HERE.md](START_HERE.md) first (short path), or **[PROJECT_COMPLETE_GUIDE.md](PROJECT_COMPLETE_GUIDE.md)** for the full architecture, production flow, thresholds, and reference in one file.

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
        └─► Groq generation with [Source: …] labels
              └─► optional faithfulness audit (ENABLE_FAITHFULNESS_GUARD) ──► answer + citations
```

When retrieval is weak, the model is instructed to say the context does not contain
enough information; the faithfulness guard can block answers that are not fully
supported by the retrieved passages (see `src/core/faithfulness_guard.py`).

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
of these choices. For a **step-by-step flow** (auth, tenants, ingest, chat, Docker),
see [PROJECT_FLOW.md](PROJECT_FLOW.md). For **contributing** (layout, lint, tests),
see [CONTRIBUTING.md](CONTRIBUTING.md).

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
| `API_KEYS` | *(empty)* | Legacy comma-separated keys (single-tenant dev). With **`DATABASE_URL` set**, auth is always on (JWT login + per-tenant `X-API-Key`). With neither Postgres nor `API_KEYS`, auth is **off** — only for isolated local dev. |
| `JWT_SECRET` | *(empty)* | Signs browser session tokens after `/auth/login`; required in production with Postgres. |
| `ENABLE_FAITHFULNESS_GUARD` | `true` | Post-generation entailment check against retrieved context. |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated browser origins. |
| `QDRANT_URL` | `http://localhost:6333` | Leave empty to use embedded on-disk Qdrant (single process only). |
| `REDIS_URL` | `redis://localhost:6379/0` | Must be a **Redis Stack** server; the plain `redis` image has no vector index. |
| `GROQ_API_KEY` / `COHERE_API_KEY` | *(empty)* | Required for generation and reranking respectively. |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `BAAI/bge-small-en-v1.5` / `384` | Validated against each other at start-up; a mismatch is a hard error. |
| `SEMANTIC_CACHE_THRESHOLD` | `0.88` | Cosine similarity required for a cache hit. Keys include **tenant id + prompt pack version** so tenants do not share entries; tune lower for stricter finance queries if you see near-duplicate hits. |

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
| `GET /tenants/me` | required | Returns the authenticated agency profile (used by the SPA login). |
| `POST /auth/register` | admin | Create a tenant; response includes the raw `api_key` **once**. |
| `POST /auth/rotate-key` | required | Tenant rotates their own API key. |
| `POST /documents/upload` | required | Queue a PDF for tenant-scoped ingestion (needs Celery worker). |
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

## Multi-tenant production (how customers get an API key)

When `DATABASE_URL` is set (Docker Compose does this by default), each **agency**
is a tenant with its own isolated corpus. Customers never share one global
`API_KEYS` value — they receive a **per-tenant secret** at onboarding.

```text
Platform admin (you)                         Agency user (customer)
        │                                              │
        ├─ Set ADMIN_API_KEY in .env                   │
        ├─ Create tenant ──► raw api_key (once) ───────┼─► Paste key in SPA login
        ├─ Ingest PDFs with --tenant-id <slug>         ├─► Chat / upload in browser
        └─ Rotate or deactivate via admin CLI/API      └─► POST /auth/rotate-key if compromised
```

**1. Create an agency (admin only)**

```bash
# CLI (uses DATABASE_URL from .env)
python scripts/manage_tenants.py create --id acme-insurance --name "Acme Insurance" --plan pro

# HTTP (requires X-Admin-Key header, not for browsers)
curl -s -X POST http://localhost:8000/auth/register \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"acme-insurance","name":"Acme Insurance","plan":"pro"}'
```

Copy the printed `api_key` immediately — it is **hashed in Postgres and cannot be
looked up again**. If it is lost, rotate it:

```bash
python scripts/manage_tenants.py rotate-key --id acme-insurance
# or: POST /auth/rotate-key with the old key still valid
```

**2. Load that tenant's documents**

```bash
docker compose --profile tools run --rm ingest --tenant-id acme-insurance --limit 10
# or upload PDFs in the SPA (Upload tab) while signed in with the agency key
```

**3. Customer uses the web app**

Open `http://localhost:8080` (or your public URL), sign in with email/password
(or the flows your deployment exposes). The SPA stores a **JWT** in `localStorage`
and sends `Authorization: Bearer …` on `/api/*` calls; server integrations should
use **`X-API-Key`** with the tenant secret instead. Retrieval and semantic cache
are scoped to the authenticated tenant.

**Security note:** JWT in `localStorage` is convenient for a demo SPA but is
vulnerable to XSS; production products often prefer httpOnly cookie sessions.

**4. Integrate from another app**

```bash
curl -s -H "X-API-Key: <agency-key>" -H "Content-Type: application/json" \
  -d '{"query":"What is delta hedging?"}' \
  http://localhost:8000/chat
```

With Postgres enabled, requests **without** a valid `Authorization: Bearer` JWT or
`X-API-Key` are rejected — there is no anonymous `default` tenant in production.

## Deployment

**Local development** (API and databases published on localhost):

```bash
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
docker compose --profile tools run --rm ingest --tenant-id your-org --limit 10
curl -fsS http://localhost:8000/health/ready
```

**Production (API not exposed; HTTPS via Caddy when `SITE_DOMAIN` is set):**

See **[DEPLOY.md](DEPLOY.md)** for the full public checklist. Quick start:

```powershell
.\scripts\deploy_production.ps1 -Domain app.example.com -Email ops@example.com -GroqKey "..." -CohereKey "..."
```

Or manually:

```bash
python scripts/setup_production_env.py --domain app.example.com --email ops@example.com --force
python scripts/verify_production_env.py
docker compose up -d --build
docker compose --profile tools up -d worker
curl -fsS https://app.example.com/api/health/ready
```

With `APP_ENV=production`, the API **refuses to start** if `JWT_SECRET`, `ADMIN_API_KEY`, or
`POSTGRES_PASSWORD` are missing/weak, if `PUBLIC_SIGNUP_ENABLED=true`, or if `CORS_ORIGINS` is
localhost-only while Postgres multi-tenancy is enabled.

The stack runs `api` (internal), `qdrant`, `redis`, `postgres`, and `frontend` (public
entry on ports 80/443). Use `docker-compose.dev.yml` only for local debugging ports.

Additional notes:

- **TLS.** Set `SITE_DOMAIN` and `CADDY_EMAIL` in `.env`; Caddy in the frontend container
  obtains Let's Encrypt certificates automatically. Leave `SITE_DOMAIN` empty for HTTP on port 80.
- **Secrets.** Inject via your platform secret store in cloud deploys; never commit `.env`.
- **Back up volumes.** `pg_data` (tenants + indexes), `qdrant_data` (vectors), `rag_data` (manifest).
- **Model warmth.** Embeddings are baked into the API image at build time.

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
cp tests/eval/golden_queries_template.json tests/eval/golden_queries.json
# fill in verified relevant_sources, then:
python tests/eval/run_eval.py --k 5
```

Reports `recall@k`, `MRR`, and the MRR lift from reranking. It runs offline against
the live index, so it is deterministic and cheap enough to run after each
ingestion. Queries whose ground truth you have not verified are reported as
skipped rather than counted.

The repository ships only the **template** file (real `golden_queries.json` is
gitignored). There are **no published benchmark numbers** in this README until you
add a verified golden set and record results here or in `PROJECT_COMPLETE_GUIDE.md`.

Hybrid weights (`ENSEMBLE_WEIGHTS`, default 0.4/0.6) and recall sizes are sensible
defaults; tune them against your golden set with `run_eval.py`.

## Troubleshooting

### Docker build: `lookup registry-1.docker.io: no such host`

This is a **network/DNS** problem on the machine (Docker cannot reach Docker Hub), not an application bug.

1. Confirm the browser can reach the internet; fix Wi‑Fi/VPN/firewall if needed.
2. In **Docker Desktop → Settings → Docker Engine**, you can set DNS, e.g. `"dns": ["8.8.8.8", "1.1.1.1"]`, then **Apply & restart**.
3. Retry: `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build`

The API `Dockerfile` no longer uses `# syntax=docker/dockerfile:1`, so BuildKit does not need an extra pull of `docker/dockerfile:1` before building. You still need Hub access for base images (`python:3.13-slim`, `node:22-alpine`, `caddy:2-alpine`, etc.).

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
reference/        optional upstream snippets (see reference/README.md); not used at runtime
```

## Attribution

The retrieval design follows patterns established by
[RAGFlow](https://github.com/infiniflow/ragflow) (Apache-2.0):

- **Layout-aware parsing** — `pymupdf4llm` uses ONNX layout/table models in the
  same *family* as RAGFlow's DeepDoc (not a byte-for-byte copy of DeepDoc).
- **Parent/child chunking** — large sections are kept for generation context while
  small child chunks are embedded and matched, so retrieval stays precise without
  starving the model of context.
- **Multiple recall** — keyword and dense candidates are fused rather than picking
  one, which matters for finance text where exact terms (ticker symbols, model
  names, Greek letters) carry meaning that embeddings blur.
- **Fused reranking** — a cross-encoder rescores the fused set before generation.

RAGFlow itself is not deployed. A small `reference/` tree holds optional upstream
snippets for comparison; this stack does not import them at runtime.

## License

[MIT](LICENSE) — Copyright (c) 2026 Harshal Zarikar.
