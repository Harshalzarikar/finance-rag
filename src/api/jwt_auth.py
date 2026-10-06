"""JWT access tokens for browser login (HS256)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from src.config.settings import Settings


def create_access_token(
    settings: Settings,
    *,
    user_id: str,
    tenant_id: str,
    email: str,
) -> tuple[str, int]:
    """Return ``(token, expires_in_seconds)``."""
    expires_in = settings.jwt_expire_minutes * 60
    now = datetime.now(UTC)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    token = jwt.encode(payload, settings.jwt_signing_key, algorithm="HS256")
    return token, expires_in


def decode_access_token(settings: Settings, token: str) -> dict[str, Any]:
    """Validate and decode a bearer token."""
    return jwt.decode(token, settings.jwt_signing_key, algorithms=["HS256"])
