"""HTTP routes."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.deps import authorize, get_pipeline, require_api_key
from src.api.health import probe_readiness
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse, SourceItem
from src.core.rag_pipeline import RAGPipeline
from src.observability.logging import get_request_id

logger = logging.getLogger(__name__)

router = APIRouter()


def _to_response(result: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        answer=result["answer"],
        sources=[SourceItem(**source) for source in result["sources"]],
        cached=result.get("cached", False),
        cache_similarity=result.get("cache_similarity"),
        request_id=get_request_id(),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    _: None = Depends(authorize),
) -> ChatResponse:
    """Answer a question from the indexed corpus."""
    try:
        # The pipeline does blocking embedding, HTTP, and LLM work — run it in a
        # worker thread so one slow query cannot stall the event loop.
        result = await anyio.to_thread.run_sync(pipeline.run, payload.query, payload.top_k_rerank)
    except Exception as exc:  # noqa: BLE001 - never leak internals to the caller
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate an answer."
        ) from exc
    return _to_response(result)


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    _: None = Depends(authorize),
) -> StreamingResponse:
    """Stream the answer as Server-Sent Events, emitting citations first."""

    def event_stream() -> Iterator[str]:
        try:
            for event in pipeline.stream(payload.query, payload.top_k_rerank):
                yield _sse(event)
        except Exception:  # noqa: BLE001 - headers are already sent, so report in-band
            logger.exception("Streaming chat request failed")
            yield _sse({"type": "error", "detail": "Failed to generate an answer."})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/health/live")
async def health_live() -> dict[str, str]:
    """Liveness: the process is running and serving requests."""
    return {"status": "ok"}


@router.get("/health/ready", response_model=HealthResponse)
async def health_ready(response: Response) -> HealthResponse:
    """Readiness: every required dependency was probed successfully."""
    ready, dependencies = await anyio.to_thread.run_sync(probe_readiness)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if ready else "degraded", dependencies=dependencies)


@router.get("/metrics", include_in_schema=False)
async def metrics(_: None = Depends(require_api_key)) -> Response:
    """Prometheus exposition endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/")
async def root() -> dict[str, Any]:
    """Service descriptor."""
    return {
        "status": "ok",
        "service": "Quantitative Finance RAG API",
        "docs": "/docs",
        "endpoints": {
            "chat": "POST /chat",
            "stream": "POST /chat/stream",
            "liveness": "GET /health/live",
            "readiness": "GET /health/ready",
        },
    }
