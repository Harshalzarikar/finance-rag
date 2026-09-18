"""Shared FastAPI dependencies."""

from __future__ import annotations

import logging
import secrets

from fastapi import Header, HTTPException, Request, Response, status

from src.api.ratelimit import get_rate_limiter
from src.config.settings import get_settings
from src.core.rag_pipeline import RAGPipeline, get_rag_pipeline

logger = logging.getLogger(__name__)

AUTH_SCHEME = "ApiKey"


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Reject the request unless it carries a configured API key.

    Authentication is only enforced when at least one key is configured; an empty
    ``API_KEYS`` value is treated as "auth disabled" so local development stays
    frictionless. Comparisons are constant-time to avoid leaking key prefixes.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
            headers={"WWW-Authenticate": AUTH_SCHEME},
        )

    if not any(secrets.compare_digest(x_api_key, candidate) for candidate in settings.api_keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": AUTH_SCHEME},
        )


def authorize(
    request: Request,
    response: Response,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Authenticate the caller, then charge the request against their rate limit.

    Combining both steps in one dependency guarantees ordering: an unauthenticated
    caller is rejected before consuming any rate-limit budget.
    """
    require_api_key(x_api_key)
    settings = get_settings()

    # Prefer the key as the identity so that users behind a shared NAT/proxy are
    # not throttled as one, falling back to the peer address when auth is off.
    identity = x_api_key or (request.client.host if request.client else "anonymous")
    result = get_rate_limiter().check(identity)

    response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_per_minute)
    response.headers["X-RateLimit-Remaining"] = str(result.remaining)

    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {settings.rate_limit_per_minute} requests per minute.",
            headers={"Retry-After": str(result.reset_seconds)},
        )


def get_pipeline() -> RAGPipeline:
    """Provide the assembled pipeline, overridable in tests via dependency_overrides."""
    return get_rag_pipeline()
