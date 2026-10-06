# Scripts

Operational CLIs for this repo. Run from the repository root unless noted.

| Script | Purpose |
|--------|---------|
| `ingest.py` | Batch-ingest PDFs into Qdrant + Postgres FTS (`--tenant-id`, `--limit`, `--reset`) |
| `manage_tenants.py` | Create tenants, users, rotate API keys, set prompt overrides |
| `setup_production_env.py` | Generate production `.env` (domain, TLS, secrets) |
| `verify_production_env.py` | Validate `.env` before deploy |
| `deploy_production.ps1` | Windows: setup + verify + `docker compose up` + worker |
| `debug_retrieval.py` | Ops: inspect retrieval/rerank per tenant (needs live stack) |
| `generate_production_secrets.py` | **Deprecated** — points to `setup_production_env.py` |

### Examples

```bash
python scripts/ingest.py --tenant-id acme-insurance --limit 10
python scripts/manage_tenants.py add-user --tenant acme-insurance --email u@acme.com --password 'Secret123!'
python scripts/setup_production_env.py --domain app.example.com --email ops@example.com --force
set PYTHONPATH=.
python scripts/debug_retrieval.py --query "summarize the paper" --tenant acme-insurance
```

Docker wrappers:

```bash
docker compose --profile tools run --rm ingest --tenant-id acme-insurance --limit 5
```
