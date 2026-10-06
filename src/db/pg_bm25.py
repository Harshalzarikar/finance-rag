"""PostgreSQL Full-Text Search (FTS) retriever — replaces ``bm25_index.pkl``.

Postgres ``to_tsvector`` + ``ts_rank`` is the production-grade equivalent of
``rank_bm25``.  It runs inside the database server, so:

- Multiple API workers share one index with no file-lock contention.
- Adding new documents is an ``INSERT``, not a full rebuild-and-serialize cycle.
- Deleting a source's chunks is a single ``DELETE WHERE source = …``.

The class implements LangChain's ``BaseRetriever`` interface, so it plugs
directly into the existing ``EnsembleRetriever`` (hybrid search) without any
changes to ``hybrid_search.py``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import sqlalchemy as sa
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from src.db.schema import ChildChunkFTS, get_session_factory

logger = logging.getLogger(__name__)

_ILIKE_STOPWORDS = frozenset(
    {"about", "detail", "details", "information", "everything", "comprehensive", "profile", "resume", "tell", "more"}
)


class PostgresBM25Retriever(BaseRetriever):
    """LangChain retriever backed by Postgres full-text search.

    Drop-in replacement for ``PersistedBM25Retriever``.
    Results are automatically scoped to ``tenant_id``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    database_url: str
    k: int = Field(8, ge=1)
    tenant_id: str = Field("default", description="Tenant to scope search results to")

    # ------------------------------------------------------------------ #
    # BaseRetriever interface                                               #
    # ------------------------------------------------------------------ #

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        del run_manager, kwargs
        if not query.strip():
            return []

        session_factory = get_session_factory(self.database_url)
        with session_factory() as session:
            # plainto_tsquery converts free-form text into a tsquery safely
            # (no special operators required from callers).
            rows = session.execute(
                sa.text(
                    """
                    SELECT content, metadata_json, source, page,
                           ts_rank(tsv, plainto_tsquery('english', :query)) AS rank
                    FROM   child_chunks_fts
                    WHERE  tsv @@ plainto_tsquery('english', :query)
                    AND    tenant_id = :tenant_id
                    ORDER  BY rank DESC
                    LIMIT  :k
                    """
                ),
                {"query": query, "k": self.k, "tenant_id": self.tenant_id},
            ).fetchall()

        documents: list[Document] = []
        for row in rows:
            meta: dict[str, Any] = {}
            if row.metadata_json:
                if isinstance(row.metadata_json, str):
                    try:
                        meta = json.loads(row.metadata_json)
                    except json.JSONDecodeError:
                        pass
                else:
                    meta = dict(row.metadata_json)

            meta["retriever"] = "postgres_fts"
            meta["bm25_score"] = float(row.rank)
            meta["source"] = row.source
            if row.page is not None:
                meta["page"] = row.page
            documents.append(Document(page_content=row.content, metadata=meta))

        if not documents:
            documents = self._fallback_name_search(query)

        logger.debug("PostgresBM25Retriever returned %d results.", len(documents))
        return documents

    def _fallback_name_search(self, query: str) -> list[Document]:
        """Match proper nouns and filenames when English FTS drops rare tokens (e.g. names)."""
        tokens = [t for t in query.lower().split() if len(t) >= 5 and t.isalpha() and t not in _ILIKE_STOPWORDS]
        if not tokens:
            return []

        # Prefer longer tokens (likely surnames) and cap how many we OR together.
        tokens = sorted(set(tokens), key=len, reverse=True)[:4]
        clauses = " OR ".join(f"content ILIKE :t{i} OR source ILIKE :t{i}" for i in range(len(tokens)))
        # Rows whose filename matches an entity token rank above body-only hits.
        source_hit = " OR ".join(f"source ILIKE :t{i}" for i in range(len(tokens)))
        params: dict[str, Any] = {"k": self.k, "tenant_id": self.tenant_id}
        for i, token in enumerate(tokens):
            params[f"t{i}"] = f"%{token}%"

        session_factory = get_session_factory(self.database_url)
        with session_factory() as session:
            rows = session.execute(
                sa.text(
                    f"""
                    SELECT content, metadata_json, source, page, 0.25 AS rank
                    FROM   child_chunks_fts
                    WHERE  tenant_id = :tenant_id
                    AND    ({clauses})
                    ORDER  BY CASE WHEN ({source_hit}) THEN 0 ELSE 1 END,
                              length(content) DESC
                    LIMIT  :k
                    """
                ),
                params,
            ).fetchall()

        documents: list[Document] = []
        for row in rows:
            meta: dict[str, Any] = {"retriever": "postgres_ilike", "bm25_score": float(row.rank)}
            meta["source"] = row.source
            if row.page is not None:
                meta["page"] = row.page
            documents.append(Document(page_content=row.content, metadata=meta))
        if documents:
            logger.info(
                "PostgresBM25Retriever ILIKE fallback returned %d results for tokens %s.",
                len(documents),
                tokens,
            )
        return documents


# ---------------------------------------------------------------------------
# Ingestion helpers (called by scripts/ingest.py and the Celery worker)
# ---------------------------------------------------------------------------


def insert_chunks(database_url: str, chunks: list[Document], tenant_id: str = "default") -> None:
    """Bulk-insert child chunks into ``child_chunks_fts`` for the given tenant."""
    if not chunks:
        return

    session_factory = get_session_factory(database_url)
    rows = []
    for chunk in chunks:
        meta = dict(chunk.metadata)
        rows.append(
            {
                "tenant_id": tenant_id,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page"),
                "metadata_json": meta,  # Pass dict directly for JSONB column
                "content": chunk.page_content,
                "tsv": sa.func.to_tsvector("english", chunk.page_content),
            }
        )

    with session_factory() as session:
        session.execute(sa.insert(ChildChunkFTS).values(rows))
        session.commit()
    logger.info("Inserted %d child chunks into Postgres FTS (tenant=%s).", len(rows), tenant_id)


def delete_chunks_by_source(database_url: str, source: str, tenant_id: str = "default") -> int:
    """Remove all child chunks belonging to *source* + *tenant_id* before re-ingestion."""
    session_factory = get_session_factory(database_url)
    with session_factory() as session:
        deleted = (
            session.query(ChildChunkFTS)
            .filter(ChildChunkFTS.source == source, ChildChunkFTS.tenant_id == tenant_id)
            .delete(synchronize_session=False)
        )
        session.commit()
    logger.debug("Deleted %d FTS rows for source '%s' (tenant=%s).", deleted, source, tenant_id)
    return deleted
