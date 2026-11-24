# Quantitative Finance RAG

A production-oriented retrieval-augmented generation (RAG) service over a corpus of
quantitative finance research papers. It parses PDFs with layout awareness, splits them into
parent/child chunks, retrieves with hybrid keyword + dense search, reranks with a
cross-encoder, generates grounded answers with citations, and caches semantically similar
queries in Redis.

The retrieval design follows patterns established by [RAGFlow](https://github.com/infiniflow/ragflow)
rather than being built on it — see [Attribution](#attribution).

## What it does

- **Layout-aware ingestion** — `pymupdf4llm` with ONNX layout/table models converts PDFs to
  per-page Markdown. Parent sections are kept for generation context; child chunks are
  embedded and matched so retrieval stays precise.
- **Hybrid retrieval** — BM25 keyword recall is fused with dense vector recall (Qdrant) using
  weighted reciprocal-rank fusion, then rescored by a Cohere cross-encoder.
- **Grounded answers** — the generator must cite passages (`[Source: …]`). Passages below a
  relevance floor are dropped, and if none clear it the service declines to answer rather than
  citing unrelated documents. An optional faithfulness guard audits the answer against the
  retrieved context before it is returned.
- **Multi-tenant** — every agency is isolated by tenant id across Postgres, Qdrant, and the
  semantic cache. JWT logins for the web app, per-tenant API keys for integrations.
- **Semantic cache** — a Redis Stack HNSW index short-circuits repeated questions. Entries are
  scoped to tenant + retrieval config + prompt version so a match can never cross tenants.
- **Operable** — readiness probes, Prometheus metrics, structured JSON logs with request IDs,
  rate limiting, and a production guard that refuses to boot with weak secrets.

## Architecture

```
                 browser
                    │  HTTPS
                    ▼
        frontend (Caddy)                SPA + TLS, reverse-proxies /api/*
                    │
                    ▼
        api (FastAPI)                   auth · rate limit · semantic cache
                                        hybrid recall · rerank · generate
        ┌──────┬──────┴──────┬──────────────┐
        ▼      ▼             ▼              ▼
     qdrant  redis       postgres        Groq / Cohere
    (vectors) (cache,    (tenants,       (generation,
              limits,     users,          reranking)
              broker)     FTS index)
```

Query path:

```
query → semantic cache ──hit──► cached answer
        │ miss
        ├─► BM25 keyword recall + Qdrant dense recall → RRF fusion
        ├─► expand child chunks to parent sections → dedup
        ├─► Cohere cross-encoder rerank (relevance floor)
        └─► Groq generation with [Source: …] labels
              └─► optional faithfulness audit ──► answer + citations
```

## Quick start

### Docker (full stack)

```bash
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
docker compose --profile tools run --rm ingest --tenant-id your-org --limit 10
curl -fsS http://localhost:8000/health/ready
```

### Local API (backing services only)

```bash
python -m venv venv
venv/Scripts/activate            # Windows  (source venv/bin/activate on Linux/macOS)
pip install -r requirements-dev.txt

docker compose up -d qdrant redis
python scripts/ingest.py --limit 5
uvicorn src.api.main:app --reload --port 8000
```

Interactive docs are at `http://localhost:8000/docs`.

## Configuration

Every setting has a default and is documented in [`.env.example`](.env.example). Environment
variables take precedence over `.env`.

| Variable | Default | Notes |
|---|---|---|
| `API_KEYS` | *(empty)* | Legacy single-tenant keys. With `DATABASE_URL` set, auth is always on (JWT + per-tenant keys). |
| `JWT_SECRET` | *(empty)* | Signs browser session tokens; required in production with Postgres. |
| `DATABASE_URL` | *(empty)* | Enables Postgres multi-tenancy (tenants, users, parent docs, FTS keyword index). |
| `GROQ_API_KEY` / `COHERE_API_KEY` | *(empty)* | Generation and reranking. Optional at boot — `/health/ready` reports `degraded`. |
| `QDRANT_URL` | `http://localhost:6333` | Empty to use embedded on-disk Qdrant (single process only). |
| `REDIS_URL` | `redis://localhost:6379/0` | Must be a Redis **Stack** server (vector index). |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `BAAI/bge-small-en-v1.5` / `384` | Validated against each other at startup. |
| `RERANK_MIN_SCORE` | `0.30` | Minimum reranker relevance for a passage to be cited; below it, the service declines to answer. |
| `SEMANTIC_CACHE_THRESHOLD` | `0.88` | Cosine similarity required for a cache hit. |
| `ENABLE_FAITHFULNESS_GUARD` | `true` | Post-generation entailment check against retrieved context. |

## API

| Endpoint | Auth | Description |
|---|---|---|
| `POST /chat` | required | Answer a question with citations. The faithfulness guard can block before the response returns. |
| `POST /chat/stream` | required | Same pipeline as Server-Sent Events (`sources` → `token` → `done`). |
| `POST /auth/login` | public | Email/password → JWT for the SPA (requires Postgres). |
| `POST /auth/signup` | public | Self-service account (gated by `PUBLIC_SIGNUP_ENABLED`). |
| `POST /auth/register` | admin | Create a tenant; returns the raw API key once. |
| `POST /documents/upload` | required | Queue a PDF for async ingestion via Celery. |
| `GET /health/live` · `GET /health/ready` | public | Liveness / readiness. |
| `GET /metrics` | required | Prometheus exposition. |

Errors return `{"detail": "..."}` with `401`, `422`, `429` (with `Retry-After`), or `500`.
Internal error text is never leaked to clients; it goes to the logs, correlated by the
`X-Request-ID` response header.

## Key engineering decisions

Three choices worth being able to explain:

1. **Tenant-scoped semantic cache.** Entries are namespaced by tenant id + retrieval config +
   prompt version, enforced as a hard RediSearch tag filter before similarity is computed, so a
   near-identical query can never be served another tenant's answer. Hits whose original query
   disagrees on numbers or reporting periods ("Q3 vs Q4") are also rejected.
2. **Hybrid recall + cross-encoder rerank.** Finance text is dense in exact terms (tickers,
   model names, Greek letters) that embeddings blur, so keyword recall is fused with dense
   recall before a cross-encoder rescores the union.
3. **Fail-closed grounding.** Sub-threshold passages are dropped and the service declines to
   answer when none clear the floor. The faithfulness guard runs a second LLM entailment audit
   before a non-cached answer is returned.

The full rationale is in [docs/architecture.md](docs/architecture.md).

## Live demo

A deployed instance is available at
[huggingface.co/spaces/zarikarharry1412/finance-rag-api](https://huggingface.co/spaces/zarikarharry1412/finance-rag-api).

## Deployment and operations

- [docs/deployment.md](docs/deployment.md) — public HTTPS checklist and production guard.
- [docs/operations.md](docs/operations.md) — migrations, backups, and monitoring.

## Development

```bash
ruff check .            # lint
ruff format --check .   # formatting
mypy                    # types
pytest -q               # tests (no network; external clients are stubbed)
```

`make check` runs all four. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Project layout

```
src/
  api/            FastAPI app, routes, auth, rate limiting, health
  cache/          Redis semantic cache
  config/         validated settings + production guard
  core/           pipeline, prompts, query/faithfulness guards
  db/             Postgres schema, tenants, users, FTS keyword index
  ingestion/      layout-aware loader, chunking, Celery tasks
  llm/            Groq generation, Cohere reranking
  observability/  structured logging, request IDs, Prometheus metrics
  retrieval/      Qdrant vector store, BM25, hybrid fusion
scripts/          ingestion, tenant management, deploy, backup
frontend/         React + Vite SPA served by Caddy
tests/            unit + API tests, offline retrieval eval
```

## Attribution

The retrieval design follows patterns established by
[RAGFlow](https://github.com/infiniflow/ragflow) (Apache-2.0): layout-aware parsing,
parent/child chunking, multi-recall fusion, and fused reranking. RAGFlow itself is not
deployed; `reference/` holds optional upstream snippets for comparison.

## License

[MIT](LICENSE) — Copyright (c) 2026 Harshal Zarikar.
