"""PostgreSQL-backed parent-document store.

Implements the same ``BaseStore[str, Document]`` interface as
``PickleFileStore`` so the existing ``ParentDocumentRetriever`` and
``RAGPipeline._resolve_parents`` need **zero changes** — they just call
``mget / mset / mdelete`` on whatever store is injected.

Why pickle inside Postgres?
  ``Document`` objects can carry arbitrary metadata. Storing them as pickle
  blobs keeps the schema simple and byte-for-byte compatible with the existing
  local store.  If you later want the content queryable, switch to JSONB.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterator, Sequence

from langchain_core.documents import Document
from langchain_core.stores import BaseStore

from src.db.schema import ParentDocument, get_session_factory

logger = logging.getLogger(__name__)


class PostgresDocStore(BaseStore[str, Document]):
    """Stores parent ``Document`` objects in the ``parent_documents`` table.

    Drop-in replacement for ``PickleFileStore``.
    All reads/writes are automatically scoped to ``tenant_id``.
    """

    def __init__(self, database_url: str, tenant_id: str = "default") -> None:
        self._Session = get_session_factory(database_url)
        self._tenant_id = tenant_id

    # ------------------------------------------------------------------
    # BaseStore interface
    # ------------------------------------------------------------------

    def mget(self, keys: Sequence[str]) -> list[Document | None]:
        if not keys:
            return []
        with self._Session() as session:
            rows = (
                session.query(ParentDocument)
                .filter(
                    ParentDocument.doc_id.in_(keys),
                    ParentDocument.tenant_id == self._tenant_id,
                )
                .all()
            )
        by_id = {row.doc_id: pickle.loads(row.content) for row in rows}  # noqa: S301
        return [by_id.get(k) for k in keys]

    def mset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        if not key_value_pairs:
            return
        with self._Session() as session:
            for doc_id, document in key_value_pairs:
                source = document.metadata.get("source", "unknown")
                blob = pickle.dumps(document)
                existing = (
                    session.query(ParentDocument)
                    .filter(
                        ParentDocument.doc_id == doc_id,
                        ParentDocument.tenant_id == self._tenant_id,
                    )
                    .first()
                )
                if existing:
                    existing.content = blob
                    existing.source = source
                else:
                    session.add(
                        ParentDocument(
                            doc_id=doc_id,
                            tenant_id=self._tenant_id,
                            source=source,
                            content=blob,
                        )
                    )
            session.commit()
        logger.debug("mset: upserted %d parent documents (tenant=%s).", len(key_value_pairs), self._tenant_id)

    def mdelete(self, keys: Sequence[str]) -> None:
        if not keys:
            return
        with self._Session() as session:
            session.query(ParentDocument).filter(
                ParentDocument.doc_id.in_(keys),
                ParentDocument.tenant_id == self._tenant_id,
            ).delete(synchronize_session=False)
            session.commit()

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        with self._Session() as session:
            query = session.query(ParentDocument.doc_id).filter(ParentDocument.tenant_id == self._tenant_id)
            if prefix:
                query = query.filter(ParentDocument.doc_id.like(f"{prefix}%"))
            for (doc_id,) in query:
                yield doc_id

    # ------------------------------------------------------------------
    # Extra helpers used by ingestion
    # ------------------------------------------------------------------

    def delete_by_source(self, source: str) -> int:
        """Remove all parent documents belonging to *source* for this tenant."""
        with self._Session() as session:
            deleted = (
                session.query(ParentDocument)
                .filter(
                    ParentDocument.source == source,
                    ParentDocument.tenant_id == self._tenant_id,
                )
                .delete(synchronize_session=False)
            )
            session.commit()
        logger.debug("delete_by_source('%s', tenant=%s): removed %d rows.", source, self._tenant_id, deleted)
        return deleted
