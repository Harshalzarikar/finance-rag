"""Cross-encoder reranking via Cohere Rerank."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import cohere
from langchain_core.documents import Document

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class CohereReranker:
    """Reranks retrieved documents, degrading gracefully when the API is unavailable."""

    def __init__(self, api_key: str | None, model: str, enabled: bool = True, timeout: int = 30) -> None:
        self.model = model
        self.client: Any | None = None

        if not enabled:
            logger.info("Reranking disabled by configuration.")
        elif not api_key:
            logger.warning("COHERE_API_KEY is not set — reranking is disabled.")
        else:
            try:
                self.client = cohere.ClientV2(api_key=api_key, timeout=timeout)
                logger.info("Cohere reranker ready (%s, timeout=%ds).", model, timeout)
            except Exception as exc:  # noqa: BLE001 - a bad key must not take the API down
                logger.error("Could not initialize the Cohere client (%s) — reranking is disabled.", exc)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def rerank(self, query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
        """Return the ``top_n`` documents most relevant to ``query``.

        Falls back to the input ordering whenever reranking cannot run, so a
        reranker outage degrades answer quality rather than availability.
        """
        if not documents:
            return []

        fallback = list(documents[:top_n])
        if self.client is None:
            return fallback

        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=[document.page_content for document in documents],
                top_n=top_n,
            )
        except Exception as exc:  # noqa: BLE001 - rerank is best-effort
            logger.error("Cohere rerank failed (%s) — returning unranked documents.", exc)
            return fallback

        reranked: list[Document] = []
        for result in response.results:
            original = documents[result.index]
            # Copy instead of mutating: these Documents are shared with the
            # retrieval layer, so in-place metadata edits leak between requests.
            reranked.append(
                Document(
                    page_content=original.page_content,
                    metadata={**original.metadata, "relevance_score": float(result.relevance_score)},
                )
            )
        logger.debug("Reranked %d → %d documents.", len(documents), len(reranked))
        return reranked


@lru_cache(maxsize=1)
def get_reranker() -> CohereReranker:
    """Return the process-wide reranker, constructing it on first use."""
    settings = get_settings()
    return CohereReranker(
        api_key=settings.cohere_api_key,
        model=settings.rerank_model,
        enabled=settings.rerank_enabled,
    )
