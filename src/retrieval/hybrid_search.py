"""Hybrid retrieval: BM25 keyword recall fused with dense vector recall.

This mirrors RAGFlow's "multiple recall" stage — keyword and dense candidates are
fused with weighted reciprocal rank fusion, then handed to the reranker.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config.settings import get_settings
from src.retrieval.bm25 import load_index
from src.retrieval.vector_store import get_retriever as get_vector_retriever

logger = logging.getLogger(__name__)


class HybridSearchService:
    """Fuses BM25 and dense retrieval, degrading to dense-only when BM25 is missing."""

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever | None = None,
        weights: tuple[float, float] = (0.4, 0.6),
    ) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

        if bm25_retriever is None:
            self.retriever: BaseRetriever = vector_retriever
            logger.warning("Keyword search unavailable — retrieval is dense-only.")
        else:
            self.retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=list(weights),
            )
            logger.info("Hybrid search enabled (bm25=%.2f, vector=%.2f).", *weights)

    @property
    def hybrid_enabled(self) -> bool:
        """True when both keyword and dense recall contribute to results."""
        return self.bm25_retriever is not None

    def search(self, query: str) -> list[Document]:
        """Return fused keyword + dense candidates for ``query``."""
        documents = self.retriever.invoke(query)
        logger.debug("Hybrid search returned %d candidates.", len(documents))
        return documents


@lru_cache(maxsize=32)
def get_hybrid_search(tenant_id: str = "default") -> HybridSearchService:
    """Build a tenant-scoped hybrid retriever.

    Keyword retriever selection:
    - ``DATABASE_URL`` set → ``PostgresBM25Retriever`` (production, scalable)
    - ``DATABASE_URL`` not set → ``PersistedBM25Retriever`` from pickle file (local dev)
    """
    settings = get_settings()

    if settings.database_url:
        from src.db.pg_bm25 import PostgresBM25Retriever

        bm25_retriever: BaseRetriever | None = PostgresBM25Retriever(
            database_url=settings.database_url,
            k=settings.bm25_k,
            tenant_id=tenant_id,
        )
        logger.info("Keyword search: PostgresBM25Retriever (tenant=%s).", tenant_id)
    else:
        bm25_retriever = load_index(settings.bm25_index_file, k=settings.bm25_k)
        if bm25_retriever is None:
            logger.warning(
                "No BM25 index at %s — run scripts/ingest.py to build one.",
                settings.bm25_index_file,
            )

    return HybridSearchService(
        vector_retriever=get_vector_retriever(tenant_id),
        bm25_retriever=bm25_retriever,
        weights=(settings.bm25_weight, settings.vector_weight),
    )
