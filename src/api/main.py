"""FastAPI application factory.

Importing this module builds the ASGI ``app`` object uvicorn serves. No heavy
dependency (embeddings, Qdrant, BM25, Redis) is constructed at import time — that
happens in the lifespan warm-up, and any failure there is reported by
``/health/ready`` rather than preventing the process from starting.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config.settings import get_settings
from src.core.rag_pipeline import get_rag_pipeline
from src.observability.logging import configure_logging
from src.observability.metrics import setup_metrics
from src.observability.middleware import RequestContextMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and warm the pipeline without preventing startup."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    logger.info(
        "Starting Quantitative Finance RAG API (model=%s, auth=%s, cache=%s, qdrant=%s)",
        settings.embedding_model,
        "on" if settings.auth_enabled else "off",
        "on" if settings.semantic_cache_enabled else "off",
        settings.qdrant_url or settings.qdrant_db_dir,
    )

    try:
        get_rag_pipeline()
        logger.info("RAG pipeline warmed up.")
    except Exception:  # noqa: BLE001 - degrade instead of refusing to serve
        logger.exception("RAG pipeline failed to initialise — /health/ready will report degraded.")

    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    """Build the ASGI application."""
    settings = get_settings()

    app = FastAPI(
        title="Quantitative Finance RAG API",
        description=(
            "Production RAG backend: DeepDoc-style layout parsing, parent/child chunking, "
            "hybrid BM25 + dense recall, cross-encoder reranking, and a Redis semantic cache."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        # Credentials are not used (auth is header-based), and combining a
        # wildcard origin with credentials is unsafe.
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

    app.include_router(router)
    setup_metrics(app)
    return app


app = create_app()
