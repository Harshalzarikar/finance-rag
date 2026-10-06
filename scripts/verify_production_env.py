#!/usr/bin/env python3
"""Validate ``.env`` before a production deploy (same checks as API startup)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# Mirror docker-compose: validate the DSN the API container will use.
pg_user = os.getenv("POSTGRES_USER", "raguser")
pg_pass = os.getenv("POSTGRES_PASSWORD", "")
pg_db = os.getenv("POSTGRES_DB", "ragdb")
if pg_pass:
    from urllib.parse import quote_plus

    os.environ["DATABASE_URL"] = (
        f"postgresql+psycopg://{quote_plus(pg_user)}:{quote_plus(pg_pass)}@postgres:5432/{quote_plus(pg_db)}"
    )

from src.config.production_guard import collect_production_errors  # noqa: E402
from src.config.settings import Settings, get_settings  # noqa: E402


def main() -> None:
    get_settings.cache_clear()
    os.environ.setdefault("APP_ENV", "production")

    try:
        settings = Settings()
    except ValueError as exc:
        print("FAIL:", exc, file=sys.stderr)
        sys.exit(1)

    errors = collect_production_errors(settings)
    if errors:
        print("Production configuration invalid:", file=sys.stderr)
        for item in errors:
            print(f"  - {item}", file=sys.stderr)
        sys.exit(1)

    print("OK: production .env passes validation.")
    print(f"   SITE_DOMAIN=https://{settings.site_domain.strip()}")
    print(f"   CORS_ORIGINS={settings.cors_origins}")
    print(f"   PUBLIC_SIGNUP_ENABLED={settings.public_signup_enabled}")


if __name__ == "__main__":
    main()
