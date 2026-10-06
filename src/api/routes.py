"""HTTP routes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator
from typing import Any

import anyio.to_thread
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.deps import TenantContext, authorize, get_pipeline, require_admin, require_api_key
from src.api.health import probe_readiness
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestJobStatus,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse,
    RotateKeyResponse,
    SessionInfo,
    SignupRequest,
    SourceItem,
    TenantAdminInfo,
    TenantInfo,
    UploadResponse,
    UserInfo,
)
from src.config.settings import get_settings
from src.core.rag_pipeline import RAGPipeline
from src.observability.logging import get_request_id

logger = logging.getLogger(__name__)

router = APIRouter()


def _login_response(user_row, settings) -> LoginResponse:
    from src.api.jwt_auth import create_access_token
    from src.db.schema import Tenant, get_session_factory

    db_url = settings.database_url
    assert db_url is not None
    Session = get_session_factory(db_url)
    with Session() as session:
        tenant_row = session.get(Tenant, user_row.tenant_id)
        if tenant_row is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Tenant not found.")
        tenant_info = TenantInfo(id=str(tenant_row.id), name=str(tenant_row.name), plan=str(tenant_row.plan))

    token, expires_in = create_access_token(
        settings,
        user_id=str(user_row.id),
        tenant_id=str(user_row.tenant_id),
        email=str(user_row.email),
    )
    return LoginResponse(
        access_token=token,
        expires_in=expires_in,
        tenant=tenant_info,
        user=UserInfo(
            id=str(user_row.id),
            email=str(user_row.email),
            name=str(user_row.full_name) if user_row.full_name else None,
        ),
    )


def _to_response(result: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        answer=result["answer"],
        sources=[SourceItem(**source) for source in result["sources"]],
        cached=result.get("cached", False),
        cache_similarity=result.get("cache_similarity"),
        request_id=get_request_id(),
        confidence_score=result.get("confidence_score"),
        faithfulness_passed=result.get("faithfulness_passed", True),
        prompt_pack_version=result.get("prompt_pack_version"),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(authorize),
) -> ChatResponse:
    """Answer a question from the tenant's indexed corpus."""
    try:
        result = await anyio.to_thread.run_sync(
            pipeline.run, payload.query, payload.top_k_rerank, [msg.model_dump() for msg in payload.chat_history]
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat request failed (tenant=%s)", tenant.id)
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
    tenant: TenantContext = Depends(authorize),
) -> StreamingResponse:
    """Stream the answer as Server-Sent Events, emitting citations first."""

    async def event_stream() -> AsyncIterator[str]:
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _produce() -> None:
            try:
                for event in pipeline.stream(
                    payload.query, payload.top_k_rerank, [msg.model_dump() for msg in payload.chat_history]
                ):
                    queue.put_nowait(_sse(event))
            except Exception:  # noqa: BLE001
                logger.exception("Streaming chat request failed (tenant=%s)", tenant.id)
                queue.put_nowait(_sse({"type": "error", "detail": "Failed to generate an answer."}))
            finally:
                queue.put_nowait(None)  # sentinel

        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _produce)

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# Document upload
# ---------------------------------------------------------------------------


@router.post("/documents/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    tenant: TenantContext = Depends(authorize),
) -> UploadResponse:
    """Upload a PDF and queue it for async ingestion via Celery.

    Returns ``202 Accepted`` immediately with a ``job_id`` to track progress.
    The document will be available for search once ingestion completes.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF files are accepted.",
        )

    # Save the upload to a temp file the Celery worker can access.
    # In production, replace this with S3/GCS: upload to object storage,
    # pass the URL to the task instead of a local path.
    suffix = f"_{file.filename}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from src.ingestion.tasks import ingest_pdf

        task = ingest_pdf.delay(
            pdf_path=tmp_path,
            source_name=file.filename,
            tenant_id=tenant.id,
        )
        logger.info("Queued ingestion job %s for tenant=%s file=%s", task.id, tenant.id, file.filename)
    except Exception as exc:  # noqa: BLE001
        os.unlink(tmp_path)
        logger.exception("Failed to queue ingestion task")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion worker unavailable. Is Redis/Celery running?",
        ) from exc

    return UploadResponse(
        job_id=task.id,
        filename=file.filename,
        tenant_id=tenant.id,
        status="queued",
        message="PDF queued for ingestion. Poll /documents/jobs/{job_id} for status.",
    )


@router.get("/documents/jobs/{job_id}", response_model=IngestJobStatus)
async def get_ingest_job(
    job_id: str,
    tenant: TenantContext = Depends(authorize),
) -> IngestJobStatus:
    """Poll the status of an async ingestion job."""
    try:
        from src.ingestion.tasks import celery_app

        task_result = celery_app.AsyncResult(job_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cannot reach the task broker.",
        ) from exc

    return IngestJobStatus(
        job_id=job_id,
        status=task_result.status,
        result=task_result.result if task_result.successful() else None,
        error=str(task_result.result) if task_result.failed() else None,
    )


# ---------------------------------------------------------------------------
# Tenant info
# ---------------------------------------------------------------------------


@router.get("/tenants/me", response_model=SessionInfo)
async def get_my_tenant(
    tenant: TenantContext = Depends(authorize),
) -> SessionInfo:
    """Return the signed-in user and their tenant workspace."""
    if not tenant.user_id or not tenant.user_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session profile requires browser login. Integrations should use X-API-Key only.",
        )
    return SessionInfo(
        tenant=TenantInfo(id=tenant.id, name=tenant.name, plan=tenant.plan),
        user=UserInfo(id=tenant.user_id, email=tenant.user_email, name=tenant.user_name),
    )


@router.post("/auth/login", response_model=LoginResponse)
async def login(payload: LoginRequest) -> LoginResponse:
    """Email/password sign-in for the web app. Returns a JWT bearer token."""
    settings = get_settings()
    if not settings.database_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Login requires DATABASE_URL (multi-tenant Postgres).",
        )

    from src.db.user_store import authenticate_user

    db_url = settings.database_url
    user_row = await anyio.to_thread.run_sync(lambda: authenticate_user(db_url, payload.email, payload.password))
    if user_row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    return _login_response(user_row, settings)


@router.post("/auth/signup", response_model=LoginResponse, status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupRequest) -> LoginResponse:
    """Create an account and sign in. Join an existing org by ID or register a new agency."""
    settings = get_settings()
    if not settings.database_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sign-up requires DATABASE_URL (multi-tenant Postgres).",
        )
    if not settings.public_signup_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public registration is disabled. Contact your administrator.",
        )

    from src.db.user_store import signup_user

    db_url = settings.database_url
    try:
        user_row = await anyio.to_thread.run_sync(
            lambda: signup_user(
                db_url,
                tenant_id=payload.tenant_id,
                email=payload.email,
                password=payload.password,
                full_name=payload.full_name,
                organization_name=payload.organization_name,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Signup failed for tenant '%s'", payload.tenant_id)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Could not create account. The email or organization ID may already be in use.",
        ) from exc

    return _login_response(user_row, settings)


# ---------------------------------------------------------------------------
# Auth routes  (admin-gated)
# ---------------------------------------------------------------------------


@router.post("/auth/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_tenant(
    payload: RegisterRequest,
    _: None = Depends(require_admin),
) -> RegisterResponse:
    """Register a new agency tenant and return their API key.

    **Admin only** — requires ``X-Admin-Key`` header.
    The raw ``api_key`` in the response is shown **exactly once** and never stored.
    Give it to the agency immediately and securely.
    """
    settings = get_settings()
    if not settings.database_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multi-tenant mode requires DATABASE_URL to be configured.",
        )
    db_url = settings.database_url
    try:
        from src.db.tenant_store import create_tenant

        raw_key, tenant_row = await anyio.to_thread.run_sync(
            lambda: create_tenant(
                db_url,
                tenant_id=payload.tenant_id,
                name=payload.name,
                plan=payload.plan,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to register tenant '%s'", payload.tenant_id)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Could not create tenant: {exc}",
        ) from exc

    return RegisterResponse(
        tenant_id=tenant_row.id,
        name=tenant_row.name,
        plan=tenant_row.plan,
        api_key=raw_key,
    )


@router.post("/auth/rotate-key", response_model=RotateKeyResponse)
async def rotate_api_key(
    tenant: TenantContext = Depends(authorize),
) -> RotateKeyResponse:
    """Rotate the calling tenant's API key. The old key is immediately invalidated."""
    settings = get_settings()
    if not settings.database_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Key rotation requires DATABASE_URL to be configured.",
        )
    db_url = settings.database_url
    try:
        from src.db.tenant_store import rotate_api_key as _rotate

        new_key = await anyio.to_thread.run_sync(lambda: _rotate(db_url, tenant.id))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Key rotation failed for tenant '%s'", tenant.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Key rotation failed.",
        ) from exc

    return RotateKeyResponse(tenant_id=tenant.id, api_key=new_key)


# ---------------------------------------------------------------------------
# Admin routes  (require X-Admin-Key)
# ---------------------------------------------------------------------------


@router.get("/admin/tenants", response_model=list[TenantAdminInfo])
async def admin_list_tenants(
    _: None = Depends(require_admin),
) -> list[TenantAdminInfo]:
    """List all registered tenants. Admin only."""
    settings = get_settings()
    if not settings.database_url:
        return []
    db_url = settings.database_url
    from src.db.tenant_store import list_tenants

    rows = await anyio.to_thread.run_sync(lambda: list_tenants(db_url))
    return [
        TenantAdminInfo(
            id=r.id,
            name=r.name,
            plan=r.plan,
            is_active=r.is_active,
            created_at=str(r.created_at),
        )
        for r in rows
    ]


@router.patch("/admin/tenants/{tenant_id}/deactivate", status_code=status.HTTP_204_NO_CONTENT)
async def admin_deactivate_tenant(
    tenant_id: str,
    _: None = Depends(require_admin),
) -> None:
    """Deactivate a tenant so their API key is immediately rejected. Admin only."""
    settings = get_settings()
    if not settings.database_url:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No database.")
    db_url = settings.database_url
    from src.db.schema import get_session_factory
    from src.db.tenant_store import get_tenant_by_id

    tenant_row = await anyio.to_thread.run_sync(lambda: get_tenant_by_id(db_url, tenant_id))
    if tenant_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tenant '{tenant_id}' not found.")
    Session = get_session_factory(db_url)

    def _deactivate():
        with Session() as s:
            row = s.get(__import__("src.db.schema", fromlist=["Tenant"]).Tenant, tenant_id)
            if row:
                row.is_active = False
                s.commit()

    await anyio.to_thread.run_sync(_deactivate)
    logger.warning("Admin deactivated tenant '%s'.", tenant_id)


@router.patch("/admin/tenants/{tenant_id}/activate", status_code=status.HTTP_204_NO_CONTENT)
async def admin_activate_tenant(
    tenant_id: str,
    _: None = Depends(require_admin),
) -> None:
    """Re-activate a previously deactivated tenant. Admin only."""
    settings = get_settings()
    if not settings.database_url:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No database.")
    db_url = settings.database_url
    from src.db.schema import Tenant, get_session_factory

    Session = get_session_factory(db_url)

    def _activate():
        with Session() as s:
            row = s.get(Tenant, tenant_id)
            if row:
                row.is_active = True
                s.commit()

    await anyio.to_thread.run_sync(_activate)
    logger.info("Admin activated tenant '%s'.", tenant_id)


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
        "service": "Policy Intelligence RAG API",
        "docs": "/docs",
        "endpoints": {
            "chat": "POST /chat",
            "stream": "POST /chat/stream",
            "upload": "POST /documents/upload",
            "job_status": "GET /documents/jobs/{job_id}",
            "tenant_me": "GET /tenants/me",
            "register": "POST /auth/register  [admin]",
            "rotate_key": "POST /auth/rotate-key",
            "admin_tenants": "GET /admin/tenants  [admin]",
            "liveness": "GET /health/live",
            "readiness": "GET /health/ready",
        },
    }
