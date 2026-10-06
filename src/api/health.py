"""Readiness probing for every downstream dependency.

Liveness (``/health/live``) only proves the process is up. Readiness
(``/health/ready``) actually exercises each dependency so that a broken Qdrant
collection or a missing BM25 index is visible instead of surfacing as a failed
user request.
"""

from __future__ import annotations

import logging

from src.api.schemas import DependencyStatus
from src.cache.semantic_cache import get_semantic_cache
from src.config.settings import get_settings
from src.core.rag_pipeline import RAGPipeline, get_rag_pipeline
from src.retrieval.vector_store import get_embeddings, get_qdrant_client, verify_collection_dim

logger = logging.getLogger(__name__)

_MAX_DETAIL = 200


def _detail(exc: Exception) -> str:
    return str(exc)[:_MAX_DETAIL]


def _check_qdrant() -> DependencyStatus:
    settings = get_settings()
    try:
        client = get_qdrant_client()
        if not client.collection_exists(settings.qdrant_collection_name):
            return DependencyStatus(
                name="qdrant",
                ok=False,
                required=True,
                detail=f"collection '{settings.qdrant_collection_name}' not found — run scripts/ingest.py",
            )
        dimension = verify_collection_dim(client)
        return DependencyStatus(
            name="qdrant",
            ok=True,
            required=True,
            detail=f"collection '{settings.qdrant_collection_name}' dim={dimension}",
        )
    except Exception as exc:  # noqa: BLE001 - health must report, never raise
        return DependencyStatus(name="qdrant", ok=False, required=True, detail=_detail(exc))


def _check_embeddings() -> DependencyStatus:
    settings = get_settings()
    try:
        vector = get_embeddings().embed_query("health probe")
        if len(vector) != settings.embedding_dim:
            return DependencyStatus(
                name="embeddings",
                ok=False,
                required=True,
                detail=f"produced {len(vector)} dimensions, configuration expects {settings.embedding_dim}",
            )
        return DependencyStatus(
            name="embeddings",
            ok=True,
            required=True,
            detail=f"{settings.embedding_model} ({len(vector)}d)",
        )
    except Exception as exc:  # noqa: BLE001
        return DependencyStatus(name="embeddings", ok=False, required=True, detail=_detail(exc))


def _check_bm25(pipeline: RAGPipeline) -> DependencyStatus:
    settings = get_settings()
    if not pipeline.retriever.hybrid_enabled:
        if settings.database_url:
            detail = "Postgres full-text index unavailable — retrieval is dense-only"
        else:
            detail = f"BM25 index missing at {settings.bm25_index_file} — retrieval is dense-only"
        return DependencyStatus(name="bm25", ok=False, required=False, detail=detail)
    if settings.database_url:
        return DependencyStatus(
            name="bm25",
            ok=True,
            required=False,
            detail="Postgres FTS keyword index (tenant-scoped)",
        )
    retriever = pipeline.retriever.bm25_retriever
    chunks = len(getattr(retriever, "corpus", []))
    return DependencyStatus(name="bm25", ok=True, required=False, detail=f"{chunks} chunks in pickle index")


def _check_redis() -> DependencyStatus:
    settings = get_settings()
    if not settings.semantic_cache_enabled:
        return DependencyStatus(name="redis", ok=True, required=False, detail="semantic cache disabled")
    if get_semantic_cache().healthy():
        return DependencyStatus(name="redis", ok=True, required=False, detail="reachable")
    return DependencyStatus(
        name="redis",
        ok=False,
        required=False,
        detail="unreachable — caching and rate limiting degrade to no-ops",
    )


def _check_reranker(pipeline: RAGPipeline) -> DependencyStatus:
    if pipeline.reranker.enabled:
        return DependencyStatus(
            name="cohere_rerank", ok=True, required=False, detail=f"model {pipeline.reranker.model}"
        )
    return DependencyStatus(
        name="cohere_rerank",
        ok=False,
        required=False,
        detail="disabled or uninitialised — results are returned unranked",
    )


def _check_generator(pipeline: RAGPipeline) -> DependencyStatus:
    model = getattr(pipeline.generator, "model_name", None) or getattr(pipeline.generator, "model", None)
    if model is None:
        return DependencyStatus(name="groq", ok=False, required=True, detail="generator has no configured model")
    return DependencyStatus(name="groq", ok=True, required=True, detail=f"client configured for {model}")


def collect_health(pipeline: RAGPipeline) -> tuple[bool, list[DependencyStatus]]:
    """Probe all dependencies and report whether the service can serve traffic."""
    statuses = [
        _check_qdrant(),
        _check_embeddings(),
        _check_generator(pipeline),
        _check_bm25(pipeline),
        _check_reranker(pipeline),
        _check_redis(),
    ]
    ready = all(status.ok for status in statuses if status.required)
    return ready, statuses


def probe_readiness() -> tuple[bool, list[DependencyStatus]]:
    """Assemble the pipeline if needed, then probe it.

    Pipeline assembly is attempted here rather than injected as a dependency so
    that a startup failure is reported as a 503 with a reason, instead of the
    readiness endpoint itself erroring out.
    """
    try:
        pipeline = get_rag_pipeline()
    except Exception as exc:  # noqa: BLE001 - this is exactly what readiness must report
        logger.error("RAG pipeline unavailable: %s", exc)
        return False, [
            DependencyStatus(name="pipeline", ok=False, required=True, detail=_detail(exc)),
        ]
    return collect_health(pipeline)
