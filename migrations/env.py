"""Alembic environment.

Reads the target database URL from, in order of precedence:

1. ``sqlalchemy.url`` on the Alembic config (set programmatically by
   ``src.db.schema.init_db``, or via ``alembic -x``).
2. ``DATABASE_URL`` in the application settings.

Migrations only make sense when PostgreSQL multi-tenancy is enabled, so a missing
URL is a hard error rather than silently pointing at a local file store.
"""

from __future__ import annotations

import logging

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.config.settings import get_settings
from src.db.schema import Base

config = context.config
target_metadata = Base.metadata

logger = logging.getLogger("alembic.env")


def _database_url() -> str:
    url = config.get_main_option("sqlalchemy.url") or ""
    if url.strip():
        return url.strip()
    settings = get_settings()
    if settings.database_url:
        return settings.database_url
    raise RuntimeError("DATABASE_URL is not set; Alembic migrations require PostgreSQL.")


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
