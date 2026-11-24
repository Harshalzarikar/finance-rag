"""Tests for the Redis semantic cache.

The cache must never be able to break a request: a Redis outage degrades to the
full retrieval path rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest
import redis

from src.cache.semantic_cache import CacheHit, SemanticCache, _scope_tag
from src.core.prompt_builder import cache_discriminators, cache_hit_is_safe


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


class _RecordingPipeline:
    def __init__(self, redis_client: _RecordingRedis) -> None:
        self._redis = redis_client

    def hset(self, key: str, mapping: dict) -> _RecordingPipeline:
        self._redis.hashes[key] = mapping
        return self

    def expire(self, key: str, ttl: int) -> _RecordingPipeline:
        self._redis.expires[key] = ttl
        return self

    def execute(self) -> list[int]:
        return [1]


class _RecordingRedis:
    """Captures the search query string and the written hash mappings."""

    def __init__(self, result: _SearchResult | None = None) -> None:
        self._result = result if result is not None else _SearchResult([])
        self.searches: list[str] = []
        self.hashes: dict[str, dict] = {}
        self.expires: dict[str, int] = {}

    def ft(self, name: str):  # noqa: ANN201, ARG002
        redis_client = self

        class _Search:
            def search(self, query, query_params=None):  # noqa: ANN001, ANN201
                del query_params
                redis_client.searches.append(query.query_string())
                return redis_client._result

        return _Search()

    def pipeline(self) -> _RecordingPipeline:
        return _RecordingPipeline(self)


def _recording_cache(client: _RecordingRedis, threshold: float = 0.88) -> SemanticCache:
    cache = SemanticCache(client, dim=4, threshold=threshold, ttl_seconds=60, max_candidates=1)
    cache._ensure_index = lambda: None  # type: ignore[method-assign]
    cache._embed = lambda text: np.ones(4, dtype=np.float32)  # type: ignore[method-assign]
    return cache


def test_lookup_is_restricted_by_scope_tag():
    client = _RecordingRedis()
    cache = _recording_cache(client)

    cache.get("explain garch", scope="tenant-a")

    assert client.searches
    assert f"@scope:{{{_scope_tag('tenant-a')}}}" in client.searches[0]


def test_same_query_in_different_scopes_writes_distinct_entries():
    client = _RecordingRedis()
    cache = _recording_cache(client)

    cache.set("explain garch", "a", [], scope="tenant-a")
    cache.set("explain garch", "a", [], scope="tenant-b")

    assert len(client.hashes) == 2
    assert all("scope" in mapping for mapping in client.hashes.values())
    assert {mapping["scope"] for mapping in client.hashes.values()} == {
        _scope_tag("tenant-a"),
        _scope_tag("tenant-b"),
    }


def test_same_query_in_the_same_scope_overwrites_one_entry():
    client = _RecordingRedis()
    cache = _recording_cache(client)

    cache.set("explain garch", "old", [], scope="tenant-a")
    cache.set("explain garch", "new", [], scope="tenant-a")

    assert len(client.hashes) == 1


def test_discriminators_are_the_specific_tokens():
    assert cache_discriminators("How did Q3 2023 revenue compare to Q4?") == {"q3", "q4", "2023"}


def test_near_duplicate_prompts_with_different_numbers_are_not_interchangeable():
    assert cache_hit_is_safe("What was Q3 revenue?", "What was Q4 revenue?") is False
    assert cache_hit_is_safe("revenue in 2023", "revenue in 2024") is False
    assert cache_hit_is_safe("revenue", "revenue in 2024") is False


def test_paraphrases_with_matching_numbers_are_interchangeable():
    assert cache_hit_is_safe("How did Q3 revenue change?", "what was the q3 revenue change") is True
    assert cache_hit_is_safe("explain garch", "what is garch") is True
