"""Request and response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from src.config.settings import get_settings


class ChatRequest(BaseModel):
    """A question to answer from the indexed corpus."""

    query: str = Field(..., min_length=1, description="Natural-language question")
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
