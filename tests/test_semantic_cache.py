"""Tests for the Redis semantic cache.

The cache must never be able to break a request: a Redis outage degrades to the
full retrieval path rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest
import redis

from src.cache.semantic_cache import CacheHit, SemanticCache


class _Doc:
    def __init__(self, **fields: str) -> None:
        self.__dict__.update(fields)


class _SearchResult:
    def __init__(self, docs: list[_Doc]) -> None:
        self.docs = docs


class _FakeSearch:
    def __init__(self, result: _SearchResult | None = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error

    def search(self, query, query_params=None):  # noqa: ANN001, ANN201
        del query, query_params
        if self._error is not None:
            raise self._error
        return self._result


class _FakeRedis:
    def __init__(self, search: _FakeSearch) -> None:
        self._search = search

    def ft(self, name: str) -> _FakeSearch:  # noqa: ARG002
        return self._search


def _cache(search: _FakeSearch | None = None, threshold: float = 0.88) -> SemanticCache:
    cache = SemanticCache(
        _FakeRedis(search) if search is not None else None,
        dim=4,
        threshold=threshold,
        ttl_seconds=60,
        max_candidates=1,
    )
    # Keep the tests off Redis entirely; we are testing hit/miss semantics.
    cache._ensure_index = lambda: None  # type: ignore[method-assign]
    cache._embed = lambda text: np.ones(4, dtype=np.float32)  # type: ignore[method-assign]
    return cache


def _hit_doc(answer: str = "a model", sources: str = '[{"source": "a.pdf"}]', score: str = "0.05") -> _Doc:
    return _Doc(query="what is garch", answer=answer, sources=sources, score=score)


def test_disabled_cache_is_a_no_op():
    cache = _cache()

    assert cache.stats.enabled is False
    assert cache.get("anything") is None
    cache.set("anything", "answer", [])
    assert cache.healthy() is False


def test_neighbour_above_threshold_is_a_hit():
    cache = _cache(_FakeSearch(_SearchResult([_hit_doc()])))

    hit = cache.get("explain garch")

    assert isinstance(hit, CacheHit)
    assert hit.answer == "a model"
    assert hit.sources == [{"source": "a.pdf"}]
    assert hit.similarity == pytest.approx(0.95)
    assert hit.matched_query == "what is garch"
    assert cache.stats.hits == 1


def test_neighbour_below_threshold_is_a_miss():
    cache = _cache(_FakeSearch(_SearchResult([_hit_doc(score="0.4")])))

    assert cache.get("explain garch") is None
    assert cache.stats.misses == 1


def test_threshold_boundary_is_inclusive():
    cache = _cache(_FakeSearch(_SearchResult([_hit_doc(score="0.12")])), threshold=0.88)

    assert cache.get("explain garch") is not None


def test_redis_failure_fails_open():
    cache = _cache(_FakeSearch(error=RuntimeError("connection refused")))

    assert cache.get("explain garch") is None
    assert cache.stats.errors == 1


def test_unreadable_sources_are_discarded():
    cache = _cache(_FakeSearch(_SearchResult([_hit_doc(sources="{not json")])))

    hit = cache.get("explain garch")

    assert hit is not None
    assert hit.sources == []


def test_empty_result_is_a_miss():
    cache = _cache(_FakeSearch(_SearchResult([])))

    assert cache.get("explain garch") is None
    assert cache.stats.misses == 1


def test_blank_query_short_circuits():
    cache = _cache(_FakeSearch(_SearchResult([_hit_doc()])))

    assert cache.get("   ") is None
    assert cache.stats.hits == 0


def test_hit_rate_is_reported():
    cache = _cache(_FakeSearch(_SearchResult([_hit_doc()])))

    cache.get("first")
    cache.get("second")

    assert cache.stats.hit_rate == 1.0
    assert cache.stats.as_dict()["hits"] == 2


def test_hit_rate_is_zero_without_traffic():
    assert _cache(_FakeSearch(_SearchResult([]))).stats.hit_rate == 0.0


class _DroppableRedis:
    """An index that can vanish, as it does on a Redis restart or FLUSHDB."""

    def __init__(self, result: _SearchResult) -> None:
        self._result = result
        self.index_exists = False
        self.recreations = 0

    def ft(self, name: str) -> _DroppableRedis:  # noqa: ARG002
        return self

    def info(self) -> dict:
        if not self.index_exists:
            raise redis.ResponseError("no such index")
        return {}

    def create_index(self, fields, definition) -> None:  # noqa: ANN001
        del fields, definition
        self.index_exists = True
        self.recreations += 1

    def search(self, query, query_params=None):  # noqa: ANN001, ANN201
        del query, query_params
        if not self.index_exists:
            raise redis.ResponseError("rag_semantic_cache: no such index")
        return self._result

    def drop(self) -> None:
        self.index_exists = False


def test_cache_recovers_when_the_index_disappears():
    """Guards a one-shot flag that left the cache dead until the API restarted."""
    client = _DroppableRedis(_SearchResult([]))
    cache = SemanticCache(client, dim=4, threshold=0.88, ttl_seconds=60, max_candidates=1)
    cache._embed = lambda text: np.ones(4, dtype=np.float32)  # type: ignore[method-assign]

    cache.get("first")
    assert client.recreations == 1

    client.drop()

    cache.get("second")
    assert client.recreations == 2
