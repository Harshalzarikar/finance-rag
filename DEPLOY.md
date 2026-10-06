# Public production deployment

Use this checklist to run the RAG app on the **public internet** with **HTTPS**, strong secrets, and no exposed API port.

For local development, see [README.md](README.md) (`docker-compose.dev.yml`).

---

## 1. Server requirements

| Requirement | Detail |
|-------------|--------|
| OS | Linux or Windows with Docker Desktop / Engine |
| RAM | 8 GB+ recommended (embeddings + Qdrant) |
| DNS | **A record** (or AAAA) pointing your hostname → server public IP |
| Firewall | Allow **TCP 80** and **443** only (do not publish API `:8000`) |
| Email | Valid address for Let's Encrypt (`CADDY_EMAIL`) |

---

## 2. One-command setup (Windows)

From the repo root:

```powershell
.\scripts\deploy_production.ps1 -Domain app.example.com -Email ops@example.com `
  -GroqKey "gsk_..." -CohereKey "..."
```

Optional **SaaS self-service signup**:

```powershell
.\scripts\deploy_production.ps1 -Domain app.example.com -Email ops@example.com -Saas
```

Skip rebuild if images are current:

```powershell
.\scripts\deploy_production.ps1 -Domain app.example.com -Email ops@example.com -SkipBuild
```

---

## 3. Manual setup (any OS)

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

Create the first customer org (save the printed API key):

```bash
python scripts/manage_tenants.py create --id acme --name "Acme Corp" --plan pro
python scripts/manage_tenants.py add-user --tenant acme --email admin@acme.com --password 'ChangeMe123!'
```

Ingest documents:

```bash
docker compose --profile tools run --rm ingest --tenant-id acme --limit 20
```

---

## 4. Verify

```bash
curl -fsS https://app.example.com/api/health/ready
curl -fsS https://app.example.com/api/health/live
```

Open **https://app.example.com** in a browser, sign in with your org id and user.

---

## 5. What production mode enforces

When `APP_ENV=production`, the API **refuses to start** unless:

- `SITE_DOMAIN` and `CADDY_EMAIL` are set  
- `GROQ_API_KEY` and `COHERE_API_KEY` are set  
- `JWT_SECRET` (≥32 chars) and `ADMIN_API_KEY` (≥24 chars)  
- `POSTGRES_PASSWORD` is not a known weak default  
- `CORS_ORIGINS` includes `https://<SITE_DOMAIN>` (added automatically if missing)  
- Signup: either `PUBLIC_SIGNUP_ENABLED=false` **or** `ALLOW_PUBLIC_SIGNUP_IN_PRODUCTION=true` (SaaS)

Traffic path:

```text
Internet → Caddy :443 (TLS) → /api/* → internal api:8000
                          → SPA static files
```

Postgres, Redis, and Qdrant are **not** published to the host in `docker-compose.yml`.

---

## 6. Operations

| Task | Command |
|------|---------|
| Logs | `docker compose logs -f api frontend` |
| Backup Postgres volume | `pg_data` Docker volume |
| Backup vectors | `qdrant_data` volume |
| Rotate admin | New `ADMIN_API_KEY` in `.env` + restart api |
| Update app | `git pull && docker compose up -d --build` |
| Prompt pack change | Edit `src/core/prompts/*.json`, rebuild api |

---

## 7. SaaS vs closed onboarding

| Mode | `.env` |
|------|--------|
| **Closed** (default) | `PUBLIC_SIGNUP_ENABLED=false` — you create tenants via `manage_tenants.py` or admin API |
| **Public signup** | Run setup with `--saas` or set `PUBLIC_SIGNUP_ENABLED=true` and `ALLOW_PUBLIC_SIGNUP_IN_PRODUCTION=true` |

---

## 8. Troubleshooting

- **Certificate errors** — DNS must resolve to this server before Caddy can issue certs; wait a few minutes after DNS propagates.  
- **502 on /api** — `docker compose ps` — api must be healthy.  
- **verify_production_env fails** — edit `.env`; ensure Groq/Cohere keys and strong `POSTGRES_PASSWORD`.  
- **Upload stuck** — start worker: `docker compose --profile tools up -d worker`.

See also [PROJECT_FLOW.md](PROJECT_FLOW.md) for architecture and tenant isolation.
