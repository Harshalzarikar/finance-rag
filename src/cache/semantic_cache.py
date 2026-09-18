"""Redis-backed semantic cache.

An incoming query is embedded with the same model used for retrieval and compared
against previously answered queries. A neighbour at or above the configured
cosine similarity short-circuits retrieval, reranking, and generation entirely.

The previous (pre-``src/`` migration) implementation kept this in a process-local
list, which meant every uvicorn worker had a cold cache. This version lives in
Redis Stack's HNSW vector index so all workers share it.

Every operation fails open: if Redis is unavailable the pipeline simply runs its
full path instead of erroring.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import redis
from redis.commands.search.field import NumericField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from src.config.settings import get_settings
from src.retrieval.vector_store import get_embeddings

logger = logging.getLogger(__name__)

INDEX_NAME = "rag_semantic_cache"
KEY_PREFIX = "rag:cache:"

# HNSW build parameters. M=16 / EF_CONSTRUCTION=200 is the standard
# recall-vs-memory trade-off for embedding-sized indexes.
_HNSW_PARAMS = {"TYPE": "FLOAT32", "DISTANCE_METRIC": "COSINE", "M": 16, "EF_CONSTRUCTION": 200}
_VECTOR_ALGORITHM = "HNSW"


@dataclass
class CacheHit:
    """A cached answer together with the query that originally produced it."""

    answer: str
    sources: list[dict[str, Any]]
    similarity: float
    matched_query: str


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    errors: int = 0
    writes: int = 0
    enabled: bool = False

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "writes": self.writes,
            "hit_rate": round(self.hit_rate, 4),
        }


def _as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


class SemanticCache:
    """Cosine-similarity cache over a Redis Stack vector index."""

    def __init__(
        self,
        client: redis.Redis | None,
        *,
        dim: int,
        threshold: float,
        ttl_seconds: int,
        max_candidates: int,
        index_name: str = INDEX_NAME,
        key_prefix: str = KEY_PREFIX,
    ) -> None:
        self.client = client
        self.dim = dim
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_candidates = max_candidates
        self.index_name = index_name
        self.key_prefix = key_prefix
        self.stats = CacheStats(enabled=client is not None)
        self._index_ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, query: str, *, retry: bool = True) -> CacheHit | None:
        """Return a cached answer for a semantically similar query, if any."""
        if self.client is None or not query.strip():
            return None

        try:
            self._ensure_index()
            response = self.client.ft(self.index_name).search(
                Query(f"*=>[KNN {self.max_candidates} @vector $vec AS score]")
                .sort_by("score")
                .return_fields("query", "answer", "sources", "score")
                .dialect(2),
                query_params={"vec": self._embed(query).tobytes()},
            )
        except redis.ResponseError as exc:
            # The index can disappear underneath a running process (Redis restart,
            # FLUSHDB). _index_ready would otherwise short-circuit the existence
            # check and leave the cache silently dead until the API restarted.
            if retry and "no such index" in str(exc).lower():
                logger.warning("Semantic cache index is missing — recreating it.")
                self._index_ready = False
                return self.get(query, retry=False)
            self.stats.errors += 1
            logger.warning("Semantic cache lookup failed (%s) — bypassing cache.", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - cache outage must not break requests
            self.stats.errors += 1
            logger.warning("Semantic cache lookup failed (%s) — bypassing cache.", exc)
            return None

        for document in response.docs:
            # RediSearch reports COSINE *distance*; convert back to similarity.
            similarity = 1.0 - float(_as_str(document.score))
            if similarity < self.threshold:
                continue

            self.stats.hits += 1
            logger.info(
                "Semantic cache HIT (similarity=%.4f) for query %r matched %r",
                similarity,
                query[:60],
                _as_str(document.query)[:60],
            )
            return CacheHit(
                answer=_as_str(document.answer),
                sources=self._decode_sources(_as_str(document.sources)),
                similarity=similarity,
                matched_query=_as_str(document.query),
            )

        self.stats.misses += 1
        return None

    def set(self, query: str, answer: str, sources: list[dict[str, Any]]) -> None:
        """Store an answer so later similar queries can reuse it."""
        if self.client is None or not query.strip():
            return

        try:
            self._ensure_index()
            key = f"{self.key_prefix}{hashlib.sha256(query.encode('utf-8')).hexdigest()[:32]}"
            now = int(time.time())
            pipeline = self.client.pipeline()
            pipeline.hset(
                key,
                mapping={
                    "query": query,
                    "vector": self._embed(query).tobytes(),
                    "answer": answer,
                    "sources": json.dumps(sources, default=str),
                    "created_at": now,
                    "expires_at": now + self.ttl_seconds,
                },
            )
            pipeline.expire(key, self.ttl_seconds)
            pipeline.execute()
            self.stats.writes += 1
        except Exception as exc:  # noqa: BLE001 - a failed write must not fail the request
            self.stats.errors += 1
            logger.warning("Semantic cache write failed (%s) — response not cached.", exc)

    def healthy(self) -> bool:
        """True when the backing Redis responds to PING."""
        if self.client is None:
            return False
        try:
            return bool(self.client.ping())
        except Exception:  # noqa: BLE001
            return False

    def index_size(self) -> int:
        """Number of cached entries, or -1 when unavailable."""
        if self.client is None:
            return -1
        try:
            return int(self.client.ft(self.index_name).info().get("num_docs", 0))
        except Exception:  # noqa: BLE001
            return -1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _embed(self, text: str) -> np.ndarray:
        vector = get_embeddings().embed_query(text)
        return np.asarray(vector, dtype=np.float32)

    def _create_index(self) -> None:
        schema = [
            VectorField(
                "vector",
                _VECTOR_ALGORITHM,
                {**_HNSW_PARAMS, "DIM": self.dim},
            ),
            TextField("query"),
            TextField("answer"),
            TextField("sources"),
            NumericField("created_at"),
            NumericField("expires_at"),
        ]
        self.client.ft(self.index_name).create_index(  # type: ignore[union-attr]
            fields=schema,
            definition=IndexDefinition(prefix=[self.key_prefix], index_type=IndexType.HASH),
        )
        logger.info("Created Redis vector index '%s' (dim=%d).", self.index_name, self.dim)

    def _ensure_index(self) -> None:
        if self._index_ready or self.client is None:
            return
        try:
            self.client.ft(self.index_name).info()
        except redis.ResponseError:
            try:
                self._create_index()
            except redis.ResponseError as exc:
                if "Index already exists" not in str(exc):
                    raise
        self._index_ready = True

    @staticmethod
    def _decode_sources(raw: str) -> list[dict[str, Any]]:
        if not raw:
            return []
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Discarding unreadable cached sources.")
            return []
        return decoded if isinstance(decoded, list) else []


@lru_cache(maxsize=1)
def get_semantic_cache() -> SemanticCache:
    """Build the cache once per process, tolerating a disabled/unreachable Redis."""
    settings = get_settings()

    client: redis.Redis | None = None
    if settings.semantic_cache_enabled:
        try:
            client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
        except Exception as exc:  # noqa: BLE001 - misconfigured URL degrades to no caching
            logger.error("Could not configure Redis from REDIS_URL (%s) — caching disabled.", exc)
            client = None
    else:
        logger.info("Semantic cache disabled by configuration.")

    return SemanticCache(
        client,
        dim=settings.embedding_dim,
        threshold=settings.semantic_cache_threshold,
        ttl_seconds=settings.semantic_cache_ttl_seconds,
        max_candidates=settings.semantic_cache_max_candidates,
    )
