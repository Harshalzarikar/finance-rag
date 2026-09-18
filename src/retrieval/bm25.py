"""Persisted BM25 keyword retrieval.

Ingestion builds a BM25 index over child chunks and persists it as a pickle of
``{"version", "bm25", "corpus", "metadatas"}``. This module owns both halves of
that contract so the tokenizer used at build time can never drift from the one
used at query time, and so keyword hits carry the same ``source``/``page``
metadata as dense hits (which is what makes citations resolvable).
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

logger = logging.getLogger(__name__)

# Bump when the on-disk layout changes.
BM25_INDEX_VERSION = 2

# Matches the tokenizer used for both indexing and querying. Lowercasing and
# splitting on non-alphanumerics keeps "Black-Scholes" and "black scholes"
# equivalent on both sides of the lookup.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Canonical tokenizer shared by index construction and querying."""
    return _TOKEN_RE.findall(text.lower())


def build_index(
    texts: Sequence[str],
    metadatas: Sequence[dict[str, Any]] | None = None,
    k: int = 8,
) -> PersistedBM25Retriever | None:
    """Build a BM25 index over ``texts``, pairing each with its metadata."""
    if not texts:
        logger.warning("No texts supplied — skipping BM25 index build.")
        return None

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 is not installed — skipping BM25 index build.")
        return None

    corpus = list(texts)
    if metadatas is None:
        metas: list[dict[str, Any]] = [{} for _ in corpus]
    else:
        metas = [dict(m) for m in metadatas]
        if len(metas) != len(corpus):
            raise ValueError(f"metadatas length {len(metas)} does not match corpus length {len(corpus)}")

    logger.info("Building BM25 index over %d chunks...", len(corpus))
    bm25 = BM25Okapi([tokenize(text) for text in corpus])
    return PersistedBM25Retriever(bm25=bm25, corpus=corpus, metadatas=metas, k=k)


def save_index(retriever: PersistedBM25Retriever, path: str) -> None:
    """Persist a built index atomically so an interrupted write cannot corrupt it."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "wb") as handle:
        pickle.dump(
            {
                "version": BM25_INDEX_VERSION,
                "bm25": retriever.bm25,
                "corpus": retriever.corpus,
                "metadatas": retriever.metadatas,
            },
            handle,
        )
    os.replace(temporary_path, path)
    logger.info("BM25 index saved → %s (%d chunks)", path, len(retriever.corpus))


def load_raw(path: str) -> dict[str, Any]:
    """Load the raw persisted payload, returning empty containers when unusable.

    Older indexes were pickled without metadata. Those are reported here so the
    caller can warn and fall back to a re-index rather than failing at query time.
    """
    if not os.path.exists(path):
        return {"corpus": [], "metadatas": []}

    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 - corrupt/unreadable index must not be fatal
        logger.warning("Could not read BM25 index at %s: %s", path, exc)
        return {"corpus": [], "metadatas": []}

    if not isinstance(payload, dict) or "corpus" not in payload:
        logger.warning(
            "BM25 index at %s has an unrecognised format (expected a dict with 'corpus'). "
            "Re-run ingestion to rebuild it.",
            path,
        )
        return {"corpus": [], "metadatas": []}

    corpus = list(payload.get("corpus") or [])
    metadatas = [dict(m) for m in (payload.get("metadatas") or [])]
    if not metadatas:
        logger.warning(
            "BM25 index at %s stores no chunk metadata; keyword citations will be unresolvable. "
            "Re-run ingestion to rebuild it.",
            path,
        )
        metadatas = [{} for _ in corpus]
    elif len(metadatas) != len(corpus):
        logger.warning(
            "BM25 index at %s is inconsistent (%d texts vs %d metadata records); ignoring metadata.",
            path,
            len(corpus),
            len(metadatas),
        )
        metadatas = [{} for _ in corpus]

    return {"corpus": corpus, "metadatas": metadatas}


def load_index(path: str, k: int = 8) -> PersistedBM25Retriever | None:
    """Load a persisted BM25 index, or ``None`` when it is missing/unusable."""
    if not os.path.exists(path):
        logger.warning("BM25 index not found at %s — falling back to pure vector search.", path)
        return None

    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 - corrupt index must not be fatal
        logger.warning("Could not load BM25 index from %s (%s) — falling back to vector search.", path, exc)
        return None

    if not isinstance(payload, dict) or "bm25" not in payload:
        logger.warning("BM25 index at %s has an unrecognised format — falling back to vector search.", path)
        return None

    corpus = list(payload.get("corpus") or [])
    metadatas = [dict(m) for m in (payload.get("metadatas") or [])]
    if not metadatas:
        logger.warning(
            "BM25 index at %s stores no chunk metadata; keyword citations will be unresolvable. "
            "Re-run ingestion to rebuild it.",
            path,
        )
        metadatas = [{} for _ in corpus]
    elif len(metadatas) != len(corpus):
        logger.warning("BM25 index at %s is inconsistent; ignoring metadata.", path)
        metadatas = [{} for _ in corpus]

    logger.info("Loaded BM25 index from %s (%d chunks).", path, len(corpus))
    return PersistedBM25Retriever(bm25=payload["bm25"], corpus=corpus, metadatas=metadatas, k=k)


class PersistedBM25Retriever(BaseRetriever):
    """A LangChain retriever backed by a pre-built ``rank_bm25`` index.

    Returning ``Document`` objects (rather than raw strings) is what allows
    :class:`~langchain_classic.retrievers.EnsembleRetriever` to fuse keyword and
    dense results, and lets the pipeline cite a real source for keyword hits.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bm25: Any = None
    corpus: list[str] = Field(default_factory=list)
    metadatas: list[dict[str, Any]] = Field(default_factory=list)
    k: int = Field(8, ge=1)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        del run_manager, kwargs
        if self.bm25 is None or not self.corpus:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        limit = min(self.k, len(self.corpus))
        top_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:limit]

        query_terms = set(query_tokens)
        documents: list[Document] = []
        for index in top_indices:
            # Gate on lexical overlap rather than on a score threshold. BM25Okapi
            # yields negative IDFs for terms appearing in most documents, which is
            # normal in a small corpus — a "score > 0" rule would silently discard
            # valid matches there.
            if not query_terms.intersection(tokenize(self.corpus[index])):
                continue
            metadata = dict(self.metadatas[index]) if index < len(self.metadatas) else {}
            metadata["retriever"] = "bm25"
            metadata["bm25_score"] = float(scores[index])
            documents.append(Document(page_content=self.corpus[index], metadata=metadata))

        logger.debug("BM25 retriever returned %d/%d candidates for query.", len(documents), limit)
        return documents
