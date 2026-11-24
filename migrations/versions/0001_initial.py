"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-11-24

Creates the four tables that previously came from ``Base.metadata.create_all`` plus
the inline ``ALTER TABLE`` statements in ``src.db.schema.init_db``. The ``tsvector``
maintenance trigger is applied as raw SQL because SQLAlchemy has no portable DDL for
it and autogenerate cannot detect it.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("api_key_hash", sa.String(), nullable=False),
        sa.Column("plan", sa.String(), nullable=False, server_default="free"),
        sa.Column("custom_instructions", sa.Text(), nullable=False, server_default=""),
        sa.Column("prompt_pack_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_unique_constraint("tenants_api_key_hash_key", "tenants", ["api_key_hash"])

    op.create_table(
        "users",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("full_name", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "parent_documents",
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False, server_default="default"),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.PrimaryKeyConstraint("doc_id"),
    )
    op.create_index("ix_parent_documents_tenant_id", "parent_documents", ["tenant_id"])
    op.create_index("ix_parent_documents_source", "parent_documents", ["source"])
    op.create_index("ix_parent_documents_tenant_source", "parent_documents", ["tenant_id", "source"])

    op.create_table(
        "child_chunks_fts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False, server_default="default"),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("metadata_json", JSONB(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tsv", TSVECTOR(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_child_chunks_fts_tsv", "child_chunks_fts", ["tsv"], postgresql_using="gin")
    op.create_index("ix_child_chunks_fts_tenant", "child_chunks_fts", ["tenant_id"])
    op.create_index("ix_child_chunks_fts_source", "child_chunks_fts", ["source"])

    # Keep the ``tsv`` column in sync with ``content`` on insert/update. This is the
    # Postgres equivalent of BM25 tokenization for English text.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION child_chunks_fts_tsv_update() RETURNS trigger AS $$
        BEGIN
            NEW.tsv := to_tsvector('english', coalesce(NEW.content, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER tsvupdate
        BEFORE INSERT OR UPDATE ON child_chunks_fts
        FOR EACH ROW EXECUTE FUNCTION child_chunks_fts_tsv_update()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS tsvupdate ON child_chunks_fts")
    op.execute("DROP FUNCTION IF EXISTS child_chunks_fts_tsv_update()")
    op.drop_table("child_chunks_fts")
    op.drop_table("parent_documents")
    op.drop_table("users")
    op.drop_table("tenants")
