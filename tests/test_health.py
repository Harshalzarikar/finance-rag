"""Tests for readiness probing.

Readiness must report a broken dependency rather than raising, and optional
dependencies (keyword index, reranker, cache) must not block serving traffic.
"""

from __future__ import annotations

from src.api import health
from src.config.settings import get_settings


class _Qdrant:
    def __init__(self, exists: bool) -> None:
        self._exists = exists

    def collection_exists(self, name: str) -> bool:
        del name
        return self._exists


class _Embeddings:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        del text
        return [0.0] * self._dimension


class _Cache:
    def __init__(self, healthy: bool) -> None:
        self._healthy = healthy

    def healthy(self) -> bool:
        return self._healthy


def _patch(
    monkeypatch,
    pipeline,
    *,
    collection_exists: bool = True,
    embedding_dimension: int = 384,
    cache_healthy: bool = True,
) -> None:
    monkeypatch.setattr(health, "get_qdrant_client", lambda: _Qdrant(collection_exists))
    monkeypatch.setattr(health, "get_embeddings", lambda: _Embeddings(embedding_dimension))
    monkeypatch.setattr(health, "get_semantic_cache", lambda: _Cache(cache_healthy))
    monkeypatch.setattr(health, "verify_collection_dim", lambda client=None: 384)
    # Keeps probe_readiness off the real pipeline (which would load the model).
    monkeypatch.setattr(health, "get_rag_pipeline", lambda: pipeline)


def _status(statuses, name):
    return next(status for status in statuses if status.name == name)


def test_ready_when_required_dependencies_are_healthy(monkeypatch, fake_pipeline):
    _patch(monkeypatch, fake_pipeline)

    ready, statuses = health.collect_health(fake_pipeline)

    assert ready is True
    assert {status.name for status in statuses} == {
        "qdrant",
        "embeddings",
        "groq",
        "bm25",
        "cohere_rerank",
        "redis",
    }


def test_missing_collection_makes_the_service_unready(monkeypatch, fake_pipeline):
    _patch(monkeypatch, fake_pipeline, collection_exists=False)

    ready, statuses = health.collect_health(fake_pipeline)

    assert ready is False
    assert _status(statuses, "qdrant").ok is False
    assert "ingest" in _status(statuses, "qdrant").detail


def test_unreachable_qdrant_is_reported_not_raised(monkeypatch, fake_pipeline):
    def boom():
        raise ConnectionError("connection refused")

    _patch(monkeypatch, fake_pipeline)
    monkeypatch.setattr(health, "get_qdrant_client", boom)

    ready, statuses = health.collect_health(fake_pipeline)

    assert ready is False
    assert _status(statuses, "qdrant").ok is False
    assert "connection refused" in _status(statuses, "qdrant").detail


def test_embedding_dimension_mismatch_makes_the_service_unready(monkeypatch, fake_pipeline):
    _patch(monkeypatch, fake_pipeline, embedding_dimension=1024)

    ready, statuses = health.collect_health(fake_pipeline)

    assert ready is False
    assert _status(statuses, "embeddings").ok is False


def test_embedding_failure_is_reported(monkeypatch, fake_pipeline):
    def boom():
        raise RuntimeError("model weights unavailable")

    _patch(monkeypatch, fake_pipeline)
    monkeypatch.setattr(health, "get_embeddings", boom)

    ready, statuses = health.collect_health(fake_pipeline)

    assert ready is False
    assert "model weights unavailable" in _status(statuses, "embeddings").detail


def test_missing_keyword_index_is_optional(monkeypatch, fake_pipeline):
    _patch(monkeypatch, fake_pipeline)
    fake_pipeline.retriever.hybrid_enabled = False

    ready, statuses = health.collect_health(fake_pipeline)

    assert ready is True
    assert _status(statuses, "bm25").ok is False
    assert _status(statuses, "bm25").required is False


def test_disabled_reranker_does_not_block_readiness(monkeypatch, fake_pipeline):
    _patch(monkeypatch, fake_pipeline)
    fake_pipeline.reranker.enabled = False

    ready, statuses = health.collect_health(fake_pipeline)

    assert ready is True
    assert _status(statuses, "cohere_rerank").ok is False
    assert _status(statuses, "cohere_rerank").required is False


def test_unreachable_redis_is_optional(monkeypatch, fake_pipeline):
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "true")
    get_settings.cache_clear()
    _patch(monkeypatch, fake_pipeline, cache_healthy=False)

    ready, statuses = health.collect_health(fake_pipeline)

    assert ready is True
    assert _status(statuses, "redis").ok is False
    assert "no-ops" in _status(statuses, "redis").detail


def test_disabled_cache_is_reported_as_healthy(monkeypatch, fake_pipeline):
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
    get_settings.cache_clear()
    _patch(monkeypatch, fake_pipeline, cache_healthy=False)

    ready, statuses = health.collect_health(fake_pipeline)

    assert ready is True
    assert _status(statuses, "redis").ok is True
    assert "disabled" in _status(statuses, "redis").detail


def test_pipeline_failure_is_reported_instead_of_raising(monkeypatch):
    def boom():
        raise RuntimeError("GROQ_API_KEY is not set")

    monkeypatch.setattr(health, "get_rag_pipeline", boom)

    ready, statuses = health.probe_readiness()

    assert ready is False
    assert statuses[0].name == "pipeline"
    assert "GROQ_API_KEY is not set" in statuses[0].detail


def test_readiness_endpoint_reports_degraded_as_503(monkeypatch, client, fake_pipeline):
    _patch(monkeypatch, fake_pipeline, collection_exists=False)

    response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert any(dependency["name"] == "qdrant" for dependency in body["dependencies"])


def test_readiness_endpoint_reports_ok_as_200(monkeypatch, client, fake_pipeline):
    _patch(monkeypatch, fake_pipeline)

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
