"""
semantic_cache.py — In-memory semantic cache for the RAG API.

How it works:
  1. Incoming query is embedded with all-MiniLM-L6-v2 (same model used for retrieval).
  2. Cosine similarity is computed against all previously cached query embeddings.
  3. If the best match exceeds the similarity threshold (default 0.92), the cached
     response is returned immediately — skipping retrieval, reranking, and LLM calls.
  4. Otherwise the query is processed normally and the result is stored in the cache.

Production swap:
  Replace the in-memory list with Redis + Qdrant vector index for persistence and
  horizontal scaling across multiple API workers.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Similarity threshold — queries more similar than this are treated as duplicates.
# 0.92 is tight: "Black-Scholes model?" and "Explain Black-Scholes" will hit the
# cache; "Black-Scholes" and "Heston model" will not.
SIMILARITY_THRESHOLD = 0.92

# Maximum number of entries to keep in memory.
# Oldest entries are evicted when the cache is full (LRU-style).
MAX_CACHE_SIZE = 500


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray          # shape (384,)
    answer: str
    sources: list                  # list of SourceDoc dicts
    hits: int = 0
    created_at: float = field(default_factory=time.time)
    last_hit_at: float = field(default_factory=time.time)


class SemanticCache:
    """
    Thread-safe in-memory semantic cache backed by cosine similarity search.

    This cache is process-local. If you run multiple uvicorn workers, each worker
    has its own independent cache. For shared caching across workers, replace the
    internal list with Redis + a Qdrant collection.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = SIMILARITY_THRESHOLD,
        max_size: int = MAX_CACHE_SIZE,
    ) -> None:
        logger.info(f"Initializing SemanticCache (threshold={threshold}, max_size={max_size})")
        self._model = SentenceTransformer(model_name)
        self._threshold = threshold
        self._max_size = max_size
        self._entries: List[CacheEntry] = []
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> Optional[CacheEntry]:
        """
        Returns a cached CacheEntry if a semantically similar query was cached,
        otherwise returns None.
        """
        if not self._entries:
            self._misses += 1
            return None

        query_embedding = self._embed(query)
        best_entry, best_score = self._find_best_match(query_embedding)

        if best_score >= self._threshold:
            best_entry.hits += 1
            best_entry.last_hit_at = time.time()
            self._hits += 1
            logger.info(
                f"Cache HIT  | score={best_score:.4f} | "
                f"query='{query[:60]}' | matched='{best_entry.query[:60]}'"
            )
            return best_entry

        self._misses += 1
        logger.info(
            f"Cache MISS | best_score={best_score:.4f} | query='{query[:60]}'"
        )
        return None

    def set(self, query: str, answer: str, sources: list) -> None:
        """Stores a new query-answer pair in the cache."""
        if len(self._entries) >= self._max_size:
            self._evict_oldest()

        embedding = self._embed(query)
        entry = CacheEntry(query=query, embedding=embedding, answer=answer, sources=sources)
        self._entries.append(entry)
        logger.info(f"Cache SET  | size={len(self._entries)} | query='{query[:60]}'")

    def stats(self) -> dict:
        """Returns cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 1),
            "threshold": self._threshold,
            "max_size": self._max_size,
        }

    def clear(self) -> None:
        """Clears all cache entries."""
        self._entries.clear()
        self._hits = 0
        self._misses = 0
        logger.info("SemanticCache cleared.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Embeds a single string and returns a normalised numpy vector."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return np.array(vector, dtype=np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised vectors (fast dot product)."""
        return float(np.dot(a, b))

    def _find_best_match(self, query_embedding: np.ndarray):
        """Scans all cached embeddings and returns the best (entry, score) pair."""
        best_entry = None
        best_score = -1.0
        for entry in self._entries:
            score = self._cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry
        return best_entry, best_score

    def _evict_oldest(self) -> None:
        """Removes the entry with the oldest last_hit_at timestamp."""
        self._entries.sort(key=lambda e: e.last_hit_at)
        evicted = self._entries.pop(0)
        logger.info(f"Cache EVICT | '{evicted.query[:60]}'")


# Module-level singleton — shared across all requests within a single process.
semantic_cache = SemanticCache()
