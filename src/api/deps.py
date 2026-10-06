"""Shared FastAPI dependencies.

``authorize()`` authenticates via **Bearer JWT** (browser login) or ``X-API-Key``
(integrations), resolves the tenant, enforces rate limits, and injects
``TenantContext`` into route handlers.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, Response, status

from src.api.jwt_auth import decode_access_token
from src.api.ratelimit import get_rate_limiter
from src.config.settings import get_settings
from src.core.rag_pipeline import RAGPipeline, get_rag_pipeline

logger = logging.getLogger(__name__)

AUTH_SCHEME = "Bearer"


# ---------------------------------------------------------------------------
# Tenant stub (used when Postgres / multi-tenancy is not configured)
# ---------------------------------------------------------------------------


@dataclass
class TenantContext:
    """Minimal tenant + optional user data injected into route handlers."""

    id: str
    name: str
    plan: str = "free"
    user_id: str | None = None
    user_email: str | None = None
    user_name: str | None = None


_DEFAULT_TENANT = TenantContext(id="default", name="Local Dev", plan="enterprise")


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _parse_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _tenant_from_jwt(token: str, settings) -> TenantContext:
    from src.db.schema import Tenant, get_session_factory

    try:
        payload = decode_access_token(settings, token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session. Sign in again.",
            headers={"WWW-Authenticate": AUTH_SCHEME},
        ) from exc

    user_id = str(payload.get("sub", ""))
    tenant_id = str(payload.get("tenant_id", ""))
    if not user_id or not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session token.",
            headers={"WWW-Authenticate": AUTH_SCHEME},
        )

    Session = get_session_factory(settings.database_url)
    with Session() as session:
        tenant_row = session.get(Tenant, tenant_id)
        if tenant_row is None or not tenant_row.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tenant is inactive or not found.",
                headers={"WWW-Authenticate": AUTH_SCHEME},
            )
        from src.db.schema import User

        user_row = session.get(User, user_id)
        if user_row is None or not user_row.is_active or str(user_row.tenant_id) != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive.",
                headers={"WWW-Authenticate": AUTH_SCHEME},
            )
        return TenantContext(
            id=str(tenant_row.id),
            name=str(tenant_row.name),
            plan=str(tenant_row.plan),
            user_id=str(user_row.id),
            user_email=str(user_row.email),
            user_name=str(user_row.full_name) if user_row.full_name else None,
        )


def _resolve_tenant_context(bearer_token: str | None, api_key: str | None) -> TenantContext:
    settings = get_settings()

    if settings.database_url:
        if bearer_token:
            return _tenant_from_jwt(bearer_token, settings)
        if api_key:
            from src.db.tenant_store import get_tenant_by_api_key

            tenant_row = get_tenant_by_api_key(settings.database_url, api_key)
            if tenant_row is None or not tenant_row.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or inactive API key.",
                    headers={"WWW-Authenticate": "ApiKey"},
                )
            return TenantContext(
                id=str(tenant_row.id),
                name=str(tenant_row.name),
                plan=str(tenant_row.plan),
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in or provide X-API-Key.",
            headers={"WWW-Authenticate": AUTH_SCHEME},
        )

    if not settings.auth_enabled:
        return _DEFAULT_TENANT

    if api_key and any(secrets.compare_digest(api_key, k) for k in settings.api_keys):
        return _DEFAULT_TENANT

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid credentials.",
        headers={"WWW-Authenticate": AUTH_SCHEME},
    )


# ---------------------------------------------------------------------------
# FastAPI dependency functions
# ---------------------------------------------------------------------------


def require_api_key(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Reject the request unless it carries valid credentials."""
    _resolve_tenant_context(_parse_bearer(authorization), x_api_key)


def authorize(
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> TenantContext:
    """Authenticate the caller, enforce rate limits, and return tenant context."""
    tenant = _resolve_tenant_context(_parse_bearer(authorization), x_api_key)
    settings = get_settings()

    if tenant.user_id:
        identity = f"user:{tenant.user_id}"
    elif x_api_key:
        identity = f"key:{x_api_key[:12]}"
    else:
        identity = request.client.host if request.client else "anonymous"

    result = get_rate_limiter().check(identity)

    response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_per_minute)
    response.headers["X-RateLimit-Remaining"] = str(result.remaining)
    response.headers["X-Tenant-ID"] = tenant.id

    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {settings.rate_limit_per_minute} requests per minute.",
            headers={"Retry-After": str(result.reset_seconds)},
        )

    return tenant


def get_pipeline(tenant: TenantContext = Depends(authorize)) -> RAGPipeline:
    """Provide a tenant-scoped pipeline, overridable in tests via dependency_overrides."""
    return get_rag_pipeline(tenant.id)


def require_admin(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> None:
    """Gate admin-only endpoints."""
    settings = get_settings()
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API is disabled. Set ADMIN_API_KEY to enable it.",
        )
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Key header.",
            headers={"WWW-Authenticate": "AdminKey"},
        )
