"""Request and response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from src.config.settings import get_settings


class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """A question to answer from the indexed corpus."""

    query: str = Field(..., min_length=1, description="Natural-language question")
    chat_history: list[ChatMessage] = Field(default_factory=list, description="Previous conversation turns")
    top_k_rerank: int = Field(
        5,
        ge=1,
        le=20,
        description="How many reranked passages to give the generator",
    )

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Query cannot be empty.")
        limit = get_settings().max_query_chars
        if len(stripped) > limit:
            raise ValueError(f"Query exceeds the maximum length of {limit} characters.")
        return stripped


class SourceItem(BaseModel):
    """A passage the answer was grounded on."""

    source: str = Field(..., description="Originating document name")
    page: int | None = Field(None, description="Page number within the document")
    score: float | None = Field(None, description="Reranker relevance score, when available")
    snippet: str = Field(..., description="Excerpt of the retrieved passage")


class ChatResponse(BaseModel):
    """A grounded answer plus its citations."""

    answer: str
    sources: list[SourceItem]
    cached: bool = Field(False, description="True when served from the semantic cache")
    cache_similarity: float | None = Field(None, description="Similarity of the matched cached query")
    request_id: str | None = Field(None, description="Correlation ID for tracing this request")
    confidence_score: float | None = Field(None, description="Answer confidence (0-1) derived from reranker scores")
    faithfulness_passed: bool = Field(
        True, description="True if the answer passed the post-generation entailment guard"
    )
    prompt_pack_version: str | None = Field(
        None, description="Prompt pack id@version used for generation (e.g. default@1.0.0)"
    )


class DependencyStatus(BaseModel):
    """Readiness of a single downstream dependency."""

    name: str
    ok: bool
    required: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    """Aggregate health of the service and everything it depends on."""

    status: str = Field(..., description="'ok' when every required dependency is healthy")
    dependencies: list[DependencyStatus]


class UploadResponse(BaseModel):
    """Returned immediately when a PDF is submitted for ingestion."""

    job_id: str = Field(..., description="Celery task ID to poll for status")
    filename: str
    tenant_id: str
    status: str = Field("queued", description="queued | processing | completed | failed")
    message: str


class IngestJobStatus(BaseModel):
    """Status of an async ingestion job."""

    job_id: str
    status: str  # PENDING | STARTED | SUCCESS | FAILURE | RETRY
    result: dict | None = None
    error: str | None = None


class TenantInfo(BaseModel):
    """Public-facing tenant metadata returned to authenticated callers."""

    id: str
    name: str
    plan: str


class UserInfo(BaseModel):
    id: str
    email: str
    name: str | None = None


class SessionInfo(BaseModel):
    """Tenant + signed-in user profile."""

    tenant: TenantInfo
    user: UserInfo


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=8, max_length=200)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    tenant: TenantInfo
    user: UserInfo


class SignupRequest(BaseModel):
    """Self-service registration: join an existing org or create a new one."""

    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=8, max_length=200)
    full_name: str = Field("", max_length=200)
    tenant_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$",
        description="Organization ID slug, e.g. acme-insurance",
    )
    organization_name: str | None = Field(
        None,
        max_length=200,
        description="Required when the organization ID does not exist yet.",
    )


# ---------------------------------------------------------------------------
# Auth / registration schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    """Admin request to onboard a new tenant."""

    tenant_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$",
        description="Unique slug, e.g. 'acme-insurance'. Lowercase letters, digits, hyphens only.",
    )
    name: str = Field(..., min_length=1, max_length=200, description="Human-readable agency name")
    plan: str = Field("free", description="Subscription plan: free | pro | enterprise")


class RegisterResponse(BaseModel):
    """Returned once when a tenant is registered. The raw API key is NOT stored."""

    tenant_id: str
    name: str
    plan: str
    api_key: str = Field(
        ...,
        description="Raw API key — shown ONCE. Store it securely; it cannot be recovered.",
    )
    message: str = Field("Tenant registered successfully. Save the api_key — it will not be shown again.")


class RotateKeyResponse(BaseModel):
    """Returned after a key rotation. The old key is immediately invalidated."""

    tenant_id: str
    api_key: str = Field(..., description="New raw API key. Old key is now invalid.")
    message: str = Field("API key rotated. Old key is immediately invalid.")


class TenantAdminInfo(BaseModel):
    """Full tenant record visible to admins."""

    id: str
    name: str
    plan: str
    is_active: bool
    created_at: str
