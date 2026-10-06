"""SQLAlchemy table definitions and engine factory.

Three tables:
- **tenants**           — one row per agency/customer; holds their hashed API key.
- **users**             — login accounts scoped to a tenant (email + password hash).
- **parent_documents**  — stores serialized parent ``Document`` objects, scoped to a tenant.
- **child_chunks_fts**  — stores child chunk text + tsvector for full-text search, scoped to a tenant.

All user-facing data is partitioned by ``tenant_id`` so Agency A can never read
Agency B's data, even if both share the same Postgres instance.
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Tenant table  (one row per customer / agency)
# ---------------------------------------------------------------------------


class Tenant(Base):
    """One row per customer (agency) that uses the SaaS.

    API keys are stored as SHA-256 hashes so a DB breach doesn't expose raw keys.
    The raw key is shown exactly once at creation time and never stored.
    """

    __tablename__ = "tenants"

    id = Column(String, primary_key=True, nullable=False)  # short slug, e.g. "acme-insurance"
    name = Column(String, nullable=False)
    # SHA-256(api_key).hexdigest() — never store the raw key.
    api_key_hash = Column(String, nullable=False, unique=True)
    plan = Column(String, nullable=False, default="free")  # free | pro | enterprise
    custom_instructions = Column(Text, nullable=False, default="")
    prompt_pack_id = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_active = Column(Boolean, nullable=False, default=True)


class User(Base):
    """Login account belonging to exactly one tenant."""

    __tablename__ = "users"

    id = Column(String, primary_key=True, nullable=False)
    tenant_id = Column(String, nullable=False, index=True)
    email = Column(String, nullable=False, unique=True, index=True)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_active = Column(Boolean, nullable=False, default=True)


class ParentDocument(Base):
    """Stores serialized LangChain ``Document`` objects for parent sections."""

    __tablename__ = "parent_documents"

    doc_id = Column(String, primary_key=True, nullable=False)
    tenant_id = Column(String, nullable=False, index=True, default="default")
    source = Column(String, nullable=False, index=True)
    content = Column(LargeBinary, nullable=False)

    __table_args__ = (Index("ix_parent_documents_tenant_source", "tenant_id", "source"),)


class ChildChunkFTS(Base):
    """Stores child chunk text and a maintained tsvector for full-text search."""

    __tablename__ = "child_chunks_fts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String, nullable=False, index=True, default="default")
    source = Column(String, nullable=False, index=True)
    page = Column(Integer, nullable=True)
    metadata_json = Column(JSONB, nullable=False, default=dict)
    content = Column(Text, nullable=False)
    tsv = Column(TSVECTOR, nullable=False)

    __table_args__ = (
        Index("ix_child_chunks_fts_tsv", "tsv", postgresql_using="gin"),
        Index("ix_child_chunks_fts_tenant", "tenant_id"),
    )


# ---------------------------------------------------------------------------
# Engine and session factory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_engine(database_url: str) -> Engine:
    """Create (and cache) a SQLAlchemy engine for the given DSN."""
    logger.info("Connecting to PostgreSQL: %s", database_url.split("@")[-1])
    return create_engine(
        database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )


@lru_cache(maxsize=1)
def get_session_factory(database_url: str):
    """Return a bound session factory."""
    return sessionmaker(bind=get_engine(database_url), expire_on_commit=False)


def init_db(database_url: str) -> None:
    """Create tables and install the tsvector trigger if they don't exist yet.

    Safe to call on every startup — ``CREATE TABLE IF NOT EXISTS`` and
    ``CREATE INDEX IF NOT EXISTS`` are idempotent in Postgres.
    """
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)

    # Idempotent migrations for columns added after the tables were first created.
    # ``create_all`` only creates missing tables; it never alters existing ones, so
    # an older volume would otherwise be missing ``tenant_id`` and fail at query time.
    migration_sql = """
    ALTER TABLE child_chunks_fts ADD COLUMN IF NOT EXISTS tenant_id VARCHAR NOT NULL DEFAULT 'default';
    CREATE INDEX IF NOT EXISTS ix_child_chunks_fts_tenant ON child_chunks_fts (tenant_id);
    ALTER TABLE parent_documents ADD COLUMN IF NOT EXISTS tenant_id VARCHAR NOT NULL DEFAULT 'default';
    CREATE INDEX IF NOT EXISTS ix_parent_documents_tenant_source ON parent_documents (tenant_id, source);
    ALTER TABLE tenants ADD COLUMN IF NOT EXISTS custom_instructions TEXT NOT NULL DEFAULT '';
    ALTER TABLE tenants ADD COLUMN IF NOT EXISTS prompt_pack_id VARCHAR NULL;
    """
    with engine.begin() as conn:
        conn.execute(__import__("sqlalchemy").text(migration_sql))

    # Install a trigger that keeps ``tsv`` in sync with ``content`` automatically.
    # ``to_tsvector('english', content)`` is the Postgres equivalent of BM25
    # tokenization for English text.
    trigger_sql = """
    CREATE OR REPLACE FUNCTION child_chunks_fts_tsv_update() RETURNS trigger AS $$
    BEGIN
        NEW.tsv := to_tsvector('english', coalesce(NEW.content, ''));
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    DROP TRIGGER IF EXISTS tsvupdate ON child_chunks_fts;

    CREATE TRIGGER tsvupdate
        BEFORE INSERT OR UPDATE ON child_chunks_fts
        FOR EACH ROW EXECUTE FUNCTION child_chunks_fts_tsv_update();
    """
    with engine.begin() as conn:
        conn.execute(__import__("sqlalchemy").text(trigger_sql))

    logger.info("Database schema is up to date.")
