"""Celery async ingestion task.

This module moves the core PDF processing logic out of the synchronous
``scripts/ingest.py`` CLI and into a Celery task that any number of workers
can consume concurrently.

Architecture
------------

  1. A trigger (API call, CLI, or CI job) enqueues ``ingest_pdf.delay(pdf_path)``.
  2. One Celery worker picks it up and runs the full parse → chunk → index cycle.
  3. Progress and errors are visible in the Celery result backend (Redis).

This is the standard pattern for SaaS-grade document processing pipelines.
``scripts/ingest.py`` is the synchronous CLI equivalent; when ``DATABASE_URL``
is set it writes the same keyword index to Postgres FTS (``child_chunks_fts``)
instead of the local pickle, mirroring this task's ``_pg_upsert``.

Broker/Backend
--------------
Both use the same Redis instance that is already running for the semantic cache.
Configure via:
    CELERY_BROKER_URL  (default: same as REDIS_URL)
    CELERY_RESULT_URL  (default: same as REDIS_URL)
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

from src.config.settings import get_settings  # noqa: E402

logger = logging.getLogger(__name__)


def _make_app() -> Celery:
    settings = get_settings()
    # Default broker/backend to Redis URL already configured for the cache.
    broker = os.environ.get("CELERY_BROKER_URL", settings.redis_url)
    backend = os.environ.get("CELERY_RESULT_URL", settings.redis_url)
    app = Celery("advance_rag", broker=broker, backend=backend)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # Retry failed tasks up to 3 times with exponential back-off.
        task_acks_late=True,
        task_reject_on_worker_lost=True,
    )
    return app


celery_app = _make_app()


# ---------------------------------------------------------------------------
# The main ingestion task
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    name="advance_rag.ingest_pdf",
    max_retries=3,
    default_retry_delay=30,
)
def ingest_pdf(self, pdf_path: str, source_name: str | None = None, tenant_id: str = "default") -> dict[str, Any]:
    """Parse, chunk, and index a single PDF for the given tenant.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        source_name: Optional display name; defaults to the file's basename.
        tenant_id: The tenant this document belongs to.

    Returns:
        A summary dict with ``pages``, ``chunks``, and ``status``.
    """
    settings = get_settings()
    name = source_name or os.path.basename(pdf_path)

    # Lazy imports: these heavy ML packages only load when the task actually
    # runs inside a worker, not at module import time.
    import redis

    from src.ingestion.deepdoc_loader import DeepDocLoader
    from src.retrieval.vector_store import delete_source, ensure_collection, get_retriever

    redis_client = redis.Redis.from_url(settings.redis_url)
    lock_name = f"ingest_lock_{tenant_id}"
    lock = redis_client.lock(lock_name, timeout=600)  # 10 minute timeout

    if not lock.acquire(blocking=False):
        logger.info("[task] Ingestion already running for tenant %s. Retrying in 30s...", tenant_id)
        raise self.retry(countdown=30)

    try:
        if not os.path.isfile(pdf_path):
            msg = (
                f"PDF not found at {pdf_path}. "
                "In Docker, uploads must be staged on the shared /data volume (not API-only /tmp). "
                "Re-upload after starting the worker with the latest API image."
            )
            logger.error("[task] %s", msg)
            raise FileNotFoundError(msg)

        # ------------------------------------------------------------------
        # 1. Parse
        # ------------------------------------------------------------------
        logger.info("[task] Parsing %s from %s", name, pdf_path)
        pages = list(DeepDocLoader(pdf_path, page_chunks=True).lazy_load())
        # The upload lands in a temp file, so the loader records a ``tmpXXXX_``
        # basename as the source. Normalise to the real filename so citations are
        # clean and re-upload/delete stays idempotent.
        for page in pages:
            page.metadata["source"] = name
            page.metadata["tenant_id"] = tenant_id
        if not pages:
            raise ValueError(f"No pages or text could be extracted from {name}. Check that the PDF is valid.")

        # ------------------------------------------------------------------
        # 2. Ensure vector collection exists, then index into Qdrant
        # ------------------------------------------------------------------
        ensure_collection()
        retriever = get_retriever(tenant_id)

        # Remove stale vectors so re-ingestion is idempotent.
        delete_source(name)
        retriever.add_documents(list(pages), ids=None)

        # ------------------------------------------------------------------
        # 3. Get the child chunks (for the keyword index)
        # ------------------------------------------------------------------
        child_chunks = _make_child_chunks(pages, retriever)

        # ------------------------------------------------------------------
        # 4. Write to Postgres (if DATABASE_URL is set)
        # ------------------------------------------------------------------
        if settings.database_url:
            _pg_upsert(settings.database_url, name, child_chunks, tenant_id)
        else:
            # Fallback: rebuild the BM25 pickle (original behavior)
            _rebuild_bm25_pickle(settings, name, child_chunks)

        result = {
            "status": "completed",
            "source": name,
            "tenant_id": tenant_id,
            "pages": len(pages),
            "chunks": len(child_chunks),
            "sha256": _file_hash(pdf_path),
        }
        logger.info("[task] Done: %s (%d pages, %d chunks, tenant=%s)", name, len(pages), len(child_chunks), tenant_id)
        return result

    except Exception as exc:  # noqa: BLE001
        logger.exception("[task] Failed to ingest %s: %s", name, exc)
        raise self.retry(exc=exc)
    finally:
        try:
            lock.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_child_chunks(pages, retriever) -> list:
    """Reproduce the child chunks using the same splitters as the retriever."""
    parents = retriever.parent_splitter.split_documents(list(pages))
    return retriever.child_splitter.split_documents(parents)


def _pg_upsert(database_url: str, source_name: str, child_chunks: list, tenant_id: str = "default") -> None:
    """Delete stale FTS rows and insert fresh ones for the given tenant."""
    from src.db.pg_bm25 import delete_chunks_by_source, insert_chunks

    delete_chunks_by_source(database_url, source_name, tenant_id=tenant_id)
    insert_chunks(database_url, child_chunks, tenant_id=tenant_id)


def _rebuild_bm25_pickle(settings: Any, source_name: str, new_chunks: list) -> None:
    """Legacy fallback: rebuild the rank_bm25 pickle file."""
    from src.retrieval import bm25

    payload = bm25.load_raw(settings.bm25_index_file)
    existing_corpus: list[str] = payload["corpus"]
    existing_metadatas: list[dict] = payload["metadatas"]

    texts: list[str] = []
    metas: list[dict] = []
    for text, meta in zip(existing_corpus, existing_metadatas, strict=False):
        if meta.get("source") != source_name:
            texts.append(text)
            metas.append(meta)

    texts.extend(c.page_content for c in new_chunks)
    metas.extend(dict(c.metadata) for c in new_chunks)

    index = bm25.build_index(texts, metas, k=settings.bm25_k)
    if index:
        bm25.save_index(index, settings.bm25_index_file)


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
