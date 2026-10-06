"""Tenant management: create, look up, and validate tenants.

Each agency is a **tenant** identified by a short slug (e.g. ``acme-insurance``).
Their API key is never stored in plain text — only its SHA-256 hash is saved so a
database breach cannot expose raw credentials.

Usage
-----
    # On-board a new agency:
    from src.db.tenant_store import create_tenant, get_tenant_by_api_key
    raw_key, tenant = create_tenant(db_url, tenant_id="acme", name="Acme Insurance")
    # raw_key is shown ONCE and then discarded.

    # On every API call:
    tenant = get_tenant_by_api_key(db_url, api_key_from_header)
    if tenant is None or not tenant.is_active:
        raise HTTPException(401)
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime

from src.db.schema import Tenant, get_session_factory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TenantPromptSettings:
    """Per-organization generation instructions (Layer D in the prompt stack)."""

    custom_instructions: str
    prompt_pack_id: str | None


def get_tenant_prompt_settings(database_url: str, tenant_id: str) -> TenantPromptSettings:
    """Load prompt overrides for a tenant; empty when the tenant row is missing."""
    Session = get_session_factory(database_url)
    with Session() as session:
        tenant = session.get(Tenant, tenant_id)
        if tenant is None:
            return TenantPromptSettings(custom_instructions="", prompt_pack_id=None)
        instructions = str(tenant.custom_instructions or "").strip()
        pack_id = str(tenant.prompt_pack_id).strip() if tenant.prompt_pack_id else None
        return TenantPromptSettings(custom_instructions=instructions, prompt_pack_id=pack_id or None)


def set_tenant_prompt_settings(
    database_url: str,
    tenant_id: str,
    *,
    custom_instructions: str | None = None,
    prompt_pack_id: str | None = None,
) -> Tenant:
    """Update org-specific prompt configuration (admin/CLI)."""
    Session = get_session_factory(database_url)
    with Session() as session:
        tenant = session.get(Tenant, tenant_id)
        if tenant is None:
            raise ValueError(f"Tenant '{tenant_id}' not found.")
        if custom_instructions is not None:
            tenant.custom_instructions = custom_instructions
        if prompt_pack_id is not None:
            tenant.prompt_pack_id = prompt_pack_id.strip() or None
        session.commit()
        session.refresh(tenant)
    logger.info("Updated prompt settings for tenant '%s'.", tenant_id)
    return tenant


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _hash_key(raw_key: str) -> str:
    """Deterministic, one-way hash of an API key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def create_tenant(
    database_url: str,
    *,
    tenant_id: str,
    name: str,
    plan: str = "free",
) -> tuple[str, Tenant]:
    """Register a new tenant and return ``(raw_api_key, tenant_row)``.

    The raw key is the *only* time it is available in plain text.
    Store it securely and share it with the customer — it cannot be recovered.
    """
    raw_key = secrets.token_urlsafe(32)
    key_hash = _hash_key(raw_key)

    Session = get_session_factory(database_url)
    with Session() as session:
        tenant = Tenant(
            id=tenant_id,
            name=name,
            api_key_hash=key_hash,
            plan=plan,
            created_at=datetime.utcnow(),
            is_active=True,
        )
        session.add(tenant)
        session.commit()
        session.refresh(tenant)

    logger.info("Created tenant '%s' (%s).", tenant_id, plan)
    return raw_key, tenant


def get_tenant_by_api_key(database_url: str, raw_key: str) -> Tenant | None:
    """Look up a tenant by their raw API key.  Returns ``None`` if not found."""
    key_hash = _hash_key(raw_key)
    Session = get_session_factory(database_url)
    with Session() as session:
        return session.query(Tenant).filter(Tenant.api_key_hash == key_hash).first()


def get_tenant_by_id(database_url: str, tenant_id: str) -> Tenant | None:
    """Fetch a tenant record by its slug ID."""
    Session = get_session_factory(database_url)
    with Session() as session:
        return session.get(Tenant, tenant_id)


def list_tenants(database_url: str) -> list[Tenant]:
    """Return all registered tenants (admin use only)."""
    Session = get_session_factory(database_url)
    with Session() as session:
        return session.query(Tenant).order_by(Tenant.created_at).all()


def rotate_api_key(database_url: str, tenant_id: str) -> str:
    """Generate and store a fresh API key for an existing tenant.

    Returns the new raw key — the old key is immediately invalidated.
    """
    raw_key = secrets.token_urlsafe(32)
    key_hash = _hash_key(raw_key)

    Session = get_session_factory(database_url)
    with Session() as session:
        tenant = session.get(Tenant, tenant_id)
        if tenant is None:
            raise ValueError(f"Tenant '{tenant_id}' not found.")
        tenant.api_key_hash = key_hash
        session.commit()

    logger.info("Rotated API key for tenant '%s'.", tenant_id)
    return raw_key
