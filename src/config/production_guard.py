"""Validate settings when ``APP_ENV=production``.

Fails fast at API startup so weak defaults (demo DB password, missing JWT secret)
never ship behind a public URL.
"""

from __future__ import annotations

import re

from src.config.settings import Settings

_KNOWN_WEAK_DB_PASSWORDS = frozenset({"ragpassword", "password", "postgres", "changeme"})
_INSECURE_JWT_FALLBACK_MARKER = "insecure-dev-only"


def collect_production_errors(settings: Settings) -> list[str]:
    """Return human-readable configuration errors (empty when OK)."""
    if settings.app_env.lower() != "production":
        return []

    errors: list[str] = []

    if not settings.site_domain.strip():
        errors.append("SITE_DOMAIN must be set to your public hostname (e.g. app.example.com).")
    if not settings.caddy_email.strip():
        errors.append("CADDY_EMAIL must be set for automatic HTTPS (Let's Encrypt contact).")

    if not settings.groq_api_key.strip():
        errors.append("GROQ_API_KEY is required in production.")
    if not settings.cohere_api_key.strip():
        errors.append("COHERE_API_KEY is required in production.")

    if settings.database_url:
        if not settings.admin_api_key or len(settings.admin_api_key.strip()) < 24:
            errors.append("ADMIN_API_KEY must be set to at least 24 characters when DATABASE_URL is enabled.")
        if not settings.jwt_secret or len(settings.jwt_secret.strip()) < 32:
            errors.append("JWT_SECRET must be at least 32 characters for browser login in production.")
        elif _INSECURE_JWT_FALLBACK_MARKER in settings.jwt_signing_key:
            errors.append("JWT_SECRET must be explicitly set (do not rely on the dev fallback).")

        db_password = _password_from_database_url(settings.database_url)
        if db_password and db_password.lower() in _KNOWN_WEAK_DB_PASSWORDS:
            errors.append(
                "DATABASE_URL uses a known weak Postgres password. Set POSTGRES_PASSWORD to a long random value."
            )

    if settings.public_signup_enabled and not settings.allow_public_signup_in_production:
        errors.append(
            "PUBLIC_SIGNUP_ENABLED is true. Set ALLOW_PUBLIC_SIGNUP_IN_PRODUCTION=true for SaaS signup, "
            "or set PUBLIC_SIGNUP_ENABLED=false for admin-only onboarding."
        )

    if settings.database_url:
        https_origins = [o for o in settings.effective_cors_origins if o.startswith("https://")]
        if not https_origins:
            errors.append(
                "CORS_ORIGINS must include your public https origin (set SITE_DOMAIN or CORS_ORIGINS explicitly)."
            )

    return errors


def _password_from_database_url(url: str) -> str | None:
    """Extract password segment from a postgresql SQLAlchemy URL."""
    match = re.match(r"^postgresql(?:\+[\w]+)?://[^:]+:([^@]+)@", url)
    if not match:
        return None
    from urllib.parse import unquote

    return unquote(match.group(1))
