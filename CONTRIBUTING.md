# Contributing

Thanks for working on this codebase. This document describes layout, conventions, and how to validate changes before opening a PR.

## Repository layout

```text
src/
  api/           FastAPI app, routes, auth, health
  config/        Settings (pydantic-settings), production guard
  core/          RAG pipeline, prompts, query/faithfulness guards
  db/            Postgres schema, tenants, users, FTS, doc store
  retrieval/     Hybrid search, Qdrant, legacy pickle BM25 (dev-only)
  ingestion/     Chunking, DeepDoc loader, Celery tasks
  llm/           Groq generator, Cohere reranker
  cache/         Redis semantic cache
  observability/ Logging, metrics, middleware

scripts/         CLI: ingest, tenants, production deploy, debug
tests/           Pytest suite (external services stubbed)
frontend/        React SPA + Caddy
reference/       Upstream RAGFlow snippets (not used at runtime)
```

**Docs:** [README.md](README.md) (quick start), [PROJECT_FLOW.md](PROJECT_FLOW.md) (end-to-end flow), [DEPLOY.md](DEPLOY.md) (public HTTPS deploy).

## Development setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt
cp .env.example .env           # fill provider keys for live tests
```

Run backing services only:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres redis qdrant
```

Run API locally:

```bash
set PYTHONPATH=.
uvicorn src.api.main:app --reload --port 8000
```

Frontend dev:

```bash
cd frontend && npm ci && npm run dev
```

## Code standards

| Tool | Purpose |
|------|---------|
| **Ruff** | Lint + import order (`pyproject.toml`) |
| **mypy** | Types on `src/` |
| **pytest** | Behaviour tests; no live Groq/Qdrant required |

Before pushing:

```bash
ruff check .
ruff format --check .
mypy
pytest -q
```

Auto-fix formatting:

```bash
ruff format .
ruff check --fix .
```

### Python style

- **Python 3.13**, type hints on public functions.
- **Settings:** use `get_settings()`; never read `os.environ` in feature code except scripts/bootstrap.
- **Logging:** module `logger = logging.getLogger(__name__)`; no `print` in `src/`.
- **Secrets:** never commit `.env`; use `.env.example` for keys only.
- **Tenants:** every user-facing data path must respect `tenant_id` from `TenantContext`.
- **Minimal diffs:** match existing patterns; avoid drive-by refactors.

### Scripts

CLI scripts may adjust `sys.path` and use `# noqa: E402` after bootstrap. Prefer `argparse` and a `main()` guard.

## Testing

- **Unit / API tests:** `pytest -q`
- **Coverage focus:** pipeline relevance floor, auth, prompt packs, production guard.
- Tests force `APP_ENV=development` and clear `DATABASE_URL` in `tests/conftest.py` so CI stays offline-friendly.

## Architecture decisions (short)

- **Postgres + Qdrant** when `DATABASE_URL` is set (production multi-tenant).
- **Pickle BM25 + local doc store** when Postgres is off (single-machine dev).
- **Dynamic prompts:** versioned JSON under `src/core/prompts/` + optional per-tenant text in DB.
- **Fail closed on bad grounding:** relevance floor and optional faithfulness guard.

## Pull requests

1. Describe **why** (not only what).
2. Run the quality commands above.
3. Update docs if behaviour, env vars, or deploy steps change.
4. Do not commit `venv/`, `.env`, caches, or generated indexes.

## Questions

See [PROJECT_FLOW.md](PROJECT_FLOW.md) for auth, ingest, and query flows. For production hosting, see [DEPLOY.md](DEPLOY.md).
