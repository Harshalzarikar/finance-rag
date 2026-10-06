"""Qdrant vector store, embeddings, and the parent/child retriever wiring.

Everything here is created through cached factories rather than at import time,
so a misconfigured or unreachable dependency surfaces as a handled startup
error (reported by ``/health/ready``) instead of an import-time crash.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, VectorParams

from src.config.settings import get_settings
from src.ingestion.chunker import get_splitters
from src.retrieval.storage import PickleFileStore

logger = logging.getLogger(__name__)


# Payload paths written by langchain-qdrant (payload is {"page_content", "metadata"}).
SOURCE_FIELD = "metadata.source"
PAGE_FIELD = "metadata.page"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the sentence-transformers embedding model once per process."""
    settings = get_settings()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing embeddings (%s on %s)...", settings.embedding_model, device)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant in server mode, or fall back to embedded local storage."""
    settings = get_settings()
    if settings.use_remote_qdrant:
        logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout,
        )

    logger.warning(
        "QDRANT_URL is not set — using embedded Qdrant at %s. "
        "Embedded mode is single-process only; use server mode for production.",
        settings.qdrant_db_dir,
    )
    return QdrantClient(path=settings.qdrant_db_dir)


def ensure_collection(client: QdrantClient | None = None) -> None:
    """Create the collection and payload indexes when they do not exist yet."""
    settings = get_settings()
    client = client or get_qdrant_client()

    if not client.collection_exists(settings.qdrant_collection_name):
        logger.info(
            "Creating collection '%s' (size=%d, distance=COSINE).",
            settings.qdrant_collection_name,
            settings.embedding_dim,
        )
        client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )

    for field_name, schema in ((SOURCE_FIELD, PayloadSchemaType.KEYWORD), (PAGE_FIELD, PayloadSchemaType.INTEGER)):
        try:
            client.create_payload_index(
                collection_name=settings.qdrant_collection_name,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception as exc:  # noqa: BLE001 - index may already exist, or local mode may not support it
            logger.debug("Payload index on %s not created: %s", field_name, exc)


def verify_collection_dim(client: QdrantClient | None = None) -> int:
    """Return the live collection's vector size, raising if it mismatches config.

    A mismatch here means queries would compare vectors from different models,
    producing meaningless scores — better to fail loudly.
    """
    settings = get_settings()
    client = client or get_qdrant_client()
    info = client.get_collection(settings.qdrant_collection_name)

    params = info.config.params.vectors
    live_dim = params.size if isinstance(params, VectorParams) else None
    if live_dim is None:
        logger.warning("Could not determine vector size for collection '%s'.", settings.qdrant_collection_name)
        return -1

    if live_dim != settings.embedding_dim:
        raise ValueError(
            f"Collection '{settings.qdrant_collection_name}' has dimension {live_dim}, but "
            f"'{settings.embedding_model}' produces {settings.embedding_dim}-dimensional vectors. "
            "Re-ingest with --reset (and point QDRANT_COLLECTION_NAME at a new collection)."
        )
    return live_dim


def count_points(client: QdrantClient | None = None) -> int:
    """Number of vectors currently stored in the configured collection."""
    settings = get_settings()
    client = client or get_qdrant_client()
    if not client.collection_exists(settings.qdrant_collection_name):
        return 0
    return int(client.count(settings.qdrant_collection_name, exact=True).count)


def delete_source(source: str, client: QdrantClient | None = None) -> None:
    """Remove every vector belonging to ``source`` so re-ingest is idempotent.

    Without this, re-running ingestion for a changed PDF would leave the previous
    chunks behind and duplicate results.
    """
    settings = get_settings()
    client = client or get_qdrant_client()
    if not client.collection_exists(settings.qdrant_collection_name):
        return

    client.delete(
        collection_name=settings.qdrant_collection_name,
        points_selector=Filter(must=[FieldCondition(key=SOURCE_FIELD, match=MatchValue(value=source))]),
    )
    logger.debug("Deleted existing vectors for source '%s'.", source)


@lru_cache(maxsize=32)
def get_docstore(tenant_id: str = "default"):
    """The parent-document store, used to expand child chunks back to full sections.

    Returns a ``PostgresDocStore`` scoped to ``tenant_id`` when ``DATABASE_URL`` is
    configured (production), or a ``PickleFileStore`` for local dev without Postgres.
    """
    settings = get_settings()
    if settings.database_url:
        from src.db.postgres_store import PostgresDocStore
        from src.db.schema import init_db

        init_db(settings.database_url)
        return PostgresDocStore(settings.database_url, tenant_id=tenant_id)

    logger.info("Using PickleFileStore (DATABASE_URL not set — local dev mode).")
    os.makedirs(settings.doc_store_dir, exist_ok=True)
    return PickleFileStore(settings.doc_store_dir)


@lru_cache(maxsize=1)
def get_vectorstore() -> QdrantVectorStore:
    """The Qdrant-backed vector store used for child-chunk embeddings."""
    settings = get_settings()
    ensure_collection()
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.qdrant_collection_name,
        embedding=get_embeddings(),
    )


@lru_cache(maxsize=32)
def get_retriever(tenant_id: str = "default") -> ParentDocumentRetriever:
    """Dense retriever: matches child chunks, returns their parent sections.

    Scoped to ``tenant_id`` via the docstore: child chunks whose parent belongs to
    another tenant resolve to ``None`` and are dropped, isolating tenants even though
    the underlying Qdrant collection is shared.
    """
    settings = get_settings()
    parent_splitter, child_splitter = get_splitters()

    logger.info(
        "Setting up ParentDocumentRetriever (tenant=%s, fetch_k=%d).",
        tenant_id,
        settings.vector_fetch_k,
    )
    return ParentDocumentRetriever(
        vectorstore=get_vectorstore(),
        docstore=get_docstore(tenant_id),
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": settings.vector_fetch_k},
    )
