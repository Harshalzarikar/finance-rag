# Deployment

Run the service on the public internet with HTTPS, strong secrets, and no exposed API port.
For local development, see [README.md](../README.md).

## Server requirements

| Requirement | Detail |
|---|---|
| OS | Linux (or Windows with Docker Desktop) |
| RAM | 8 GB+ (embeddings + Qdrant) |
| DNS | A/AAAA record pointing the hostname at the server's public IP |
| Firewall | Allow TCP 80 and 443 only; do not publish the API `:8000` |
| Email | Valid address for Let's Encrypt (`CADDY_EMAIL`) |

## One-command setup (Windows)

```powershell
.\scripts\deploy_production.ps1 -Domain app.example.com -Email ops@example.com `
  -GroqKey "gsk_..." -CohereKey "..."
```

Optional SaaS self-service signup: add `-Saas`. Skip the rebuild with `-SkipBuild`.

## Manual setup

```bash
python scripts/setup_production_env.py \
  --domain app.example.com \
  --email ops@example.com \
  --groq-key "$GROQ_API_KEY" \
  --cohere-key "$COHERE_API_KEY" \
  --force

python scripts/verify_production_env.py
docker compose up -d --build
docker compose --profile tools up -d worker
```

Create the first tenant (save the printed API key):

```bash
python scripts/manage_tenants.py create --id acme --name "Acme Corp" --plan pro
python scripts/manage_tenants.py add-user --tenant acme --email admin@acme.com --password 'ChangeMe123!'
```

Ingest documents:

```bash
docker compose --profile tools run --rm ingest --tenant-id acme --limit 20
```

## Verify

```bash
curl -fsS https://app.example.com/api/health/ready
curl -fsS https://app.example.com/api/health/live
```

## What production mode enforces

With `APP_ENV=production`, the API refuses to start unless:

- `SITE_DOMAIN` and `CADDY_EMAIL` are set
- `GROQ_API_KEY` and `COHERE_API_KEY` are set
- `JWT_SECRET` (≥32 chars) and `ADMIN_API_KEY` (≥24 chars) are set
- `POSTGRES_PASSWORD` is not a known weak default
- `CORS_ORIGINS` includes `https://<SITE_DOMAIN>` (added automatically if missing)
- Either `PUBLIC_SIGNUP_ENABLED=false` or `ALLOW_PUBLIC_SIGNUP_IN_PRODUCTION=true`

Traffic path:

```text
Internet → Caddy :443 (TLS) → /api/* → internal api:8000
                          → SPA static files
```

Postgres, Redis, and Qdrant are not published to the host.

## SaaS vs closed onboarding

| Mode | Configuration |
|---|---|
| Closed (default) | `PUBLIC_SIGNUP_ENABLED=false` — tenants created via `manage_tenants.py` or the admin API |
| Public signup | `PUBLIC_SIGNUP_ENABLED=true` and `ALLOW_PUBLIC_SIGNUP_IN_PRODUCTION=true` |

## Releasing

Tagging a version runs [`release.yml`](../.github/workflows/release.yml), which builds and
publishes the API and frontend images to GHCR (`ghcr.io/harshalzarikar/finance-rag-api`
and `-frontend`):

```bash
git tag v2.1.0 && git push origin v2.1.0
```

`docker-compose.yml` builds images locally, which is fine for a single server; point the
`image:` references at GHCR if you deploy to a fleet that pulls from a registry.

## Troubleshooting

- **Certificate errors** — DNS must resolve before Caddy can issue certificates; allow a few
  minutes after propagation.
- **502 on /api** — `docker compose ps`; the api service must be healthy.
- **verify_production_env fails** — check Groq/Cohere keys and `POSTGRES_PASSWORD`.
- **Upload stuck** — start the worker: `docker compose --profile tools up -d worker`.
