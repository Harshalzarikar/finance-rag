"""Tenant-scoped user accounts for email/password login."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

import bcrypt

from src.db.schema import Tenant, User, get_session_factory

logger = logging.getLogger(__name__)


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_user(
    database_url: str,
    *,
    tenant_id: str,
    email: str,
    password: str,
    full_name: str = "",
) -> User:
    """Register a login for an existing tenant. Email must be unique globally."""
    normalized = email.strip().lower()
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")

    Session = get_session_factory(database_url)
    with Session() as session:
        tenant = session.get(Tenant, tenant_id)
        if tenant is None or not tenant.is_active:
            raise ValueError(f"Tenant '{tenant_id}' not found or inactive.")

        existing = session.query(User).filter(User.email == normalized).first()
        if existing is not None:
            raise ValueError(f"Email '{normalized}' is already registered.")

        user = User(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            email=normalized,
            password_hash=_hash_password(password),
            full_name=full_name.strip() or None,
            created_at=datetime.utcnow(),
            is_active=True,
        )
        session.add(user)
        session.commit()
        session.refresh(user)

    logger.info("Created user %s for tenant '%s'.", normalized, tenant_id)
    return user


def authenticate_user(database_url: str, email: str, password: str) -> User | None:
    """Return the user row when credentials are valid and tenant/user are active."""
    normalized = email.strip().lower()
    Session = get_session_factory(database_url)
    with Session() as session:
        user = session.query(User).filter(User.email == normalized).first()
        if user is None or not user.is_active:
            return None
        if not _verify_password(password, str(user.password_hash)):
            return None
        tenant = session.get(Tenant, user.tenant_id)
        if tenant is None or not tenant.is_active:
            return None
        return user


def get_user_by_id(database_url: str, user_id: str) -> User | None:
    Session = get_session_factory(database_url)
    with Session() as session:
        return session.get(User, user_id)


def list_users_for_tenant(database_url: str, tenant_id: str) -> list[User]:
    Session = get_session_factory(database_url)
    with Session() as session:
        return session.query(User).filter(User.tenant_id == tenant_id).order_by(User.created_at).all()


def signup_user(
    database_url: str,
    *,
    tenant_id: str,
    email: str,
    password: str,
    full_name: str = "",
    organization_name: str | None = None,
) -> User:
    """Join an existing tenant or create a new tenant + first user."""
    Session = get_session_factory(database_url)
    with Session() as session:
        tenant = session.get(Tenant, tenant_id)

    if tenant is None:
        if not organization_name or not organization_name.strip():
            raise ValueError("No organization exists with that ID. Enter your agency name to register a new workspace.")
        from src.db.tenant_store import create_tenant

        create_tenant(
            database_url,
            tenant_id=tenant_id,
            name=organization_name.strip(),
            plan="free",
        )
        logger.info("Self-service signup created tenant '%s'.", tenant_id)
    elif not tenant.is_active:
        raise ValueError(f"Organization '{tenant_id}' is not accepting sign-ups.")

    return create_user(
        database_url,
        tenant_id=tenant_id,
        email=email,
        password=password,
        full_name=full_name,
    )
