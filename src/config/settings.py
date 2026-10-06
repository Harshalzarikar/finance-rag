"""Central application configuration.

Values are read from environment variables and the ``.env`` file. This module
intentionally exposes :func:`get_settings` (a cached factory) instead of a
module-level singleton, so that importing any module stays side-effect free and
tests can swap configuration via ``get_settings.cache_clear()``.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated, Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

logger = logging.getLogger(__name__)

# Known embedding models and their vector dimensions. Used to catch the
# collection-dimension drift that silently corrupts a Qdrant index.
EMBEDDING_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
}

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _split_csv(value: Any) -> list[str]:
    """Parse ``a,b,c`` (or a JSON list) into a list of stripped strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value).strip()]


def _split_floats(value: Any) -> list[float]:
    """Parse ``0.4,0.6`` (or a JSON list) into a list of floats."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if isinstance(value, str):
        return [float(part) for part in value.replace(",", " ").split() if part]
    return [float(value)]


class Settings(BaseSettings):
    """Validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Credentials
    #
    # These are optional here on purpose: ingestion and retrieval must work
    # without them. When a key is missing the corresponding client fails to
    # initialise and /health/ready reports the service as degraded, which is
    # more observable than crashing the container on start.
    # ------------------------------------------------------------------
    groq_api_key: str = Field("", description="API key for Groq generation")
    groq_model_name: str = Field("openai/gpt-oss-120b", description="Groq model used for generation")
    groq_request_timeout: int = Field(60, ge=1, description="Groq LLM request timeout in seconds")
    cohere_api_key: str = Field("", description="API key for Cohere Rerank")
    cohere_timeout: int = Field(30, ge=1, description="Cohere reranker request timeout in seconds")

    # ------------------------------------------------------------------
    # API security
    # ------------------------------------------------------------------
    api_keys: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Accepted X-API-Key values, comma separated. Empty disables authentication.",
    )
    # Separate high-privilege key that gates /auth/register and /admin/* routes.
    # Set ADMIN_API_KEY in .env or the environment. If empty, admin routes are disabled.
    admin_api_key: str = Field(
        "",
        description="Secret key required to register new tenants and call admin endpoints.",
    )
    jwt_secret: str = Field(
        "",
        description="HS256 signing key for login JWTs. Set a long random value in production.",
    )
    jwt_expire_minutes: int = Field(60 * 24, ge=5, description="Browser session lifetime in minutes")
    public_signup_enabled: bool = Field(
        True,
        description="When true, users can self-register via POST /auth/signup.",
    )
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"],
        description="Allowed CORS origins, comma separated.",
    )
    rate_limit_per_minute: int = Field(30, ge=1, description="Per-caller request budget per minute")
    max_query_chars: int = Field(2000, ge=1, description="Maximum accepted query length")
    app_env: str = Field(
        "development",
        description="Set to 'production' to enforce strong secrets and disable unsafe defaults.",
    )
    site_domain: str = Field(
        "",
        description="Public hostname (no scheme). Used for TLS via Caddy and CORS alignment.",
    )
    caddy_email: str = Field("", description="Let's Encrypt contact email for the frontend Caddy container.")
    allow_public_signup_in_production: bool = Field(
        False,
        description="When true with PUBLIC_SIGNUP_ENABLED, allows self-service signup in APP_ENV=production.",
    )

    # ------------------------------------------------------------------
    # Vector database
    # ------------------------------------------------------------------
    qdrant_db_dir: str = Field("./qdrant_db_local", description="Path to embedded Qdrant storage")
    qdrant_collection_name: str = Field("parent_document_store", description="Qdrant collection name")
    qdrant_url: str | None = Field(None, description="Remote Qdrant URL. When set, server mode is used.")
    qdrant_api_key: str | None = Field(None, description="Qdrant API key")
    qdrant_timeout: int = Field(120, ge=1, description="Qdrant request timeout in seconds")

    # ------------------------------------------------------------------
    # Local storage
    # ------------------------------------------------------------------
    doc_store_dir: str = Field("./doc_store_local", description="Path to the parent document store")
    raw_pdfs_dir: str = Field("./real_pdfs", description="Directory containing source PDFs")
    bm25_index_file: str = Field("bm25_index.pkl", description="Persisted BM25 index path")
    ingestion_manifest_file: str = Field("ingestion_manifest.json", description="Ingestion manifest path")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    embedding_model: str = Field(DEFAULT_EMBEDDING_MODEL, description="Sentence-Transformers model id")
    embedding_dim: int = Field(384, ge=1, description="Vector dimension of the embedding model")

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    parent_chunk_size: int = Field(4000, ge=1, description="Maximum parent (section) chunk size in characters")
    child_chunk_size: int = Field(700, ge=1, description="Maximum child chunk size in characters")
    child_chunk_overlap: int = Field(80, ge=0, description="Child chunk overlap in characters")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    bm25_k: int = Field(8, ge=1, description="Documents recalled by BM25")
    vector_k: int = Field(8, ge=1, description="Documents recalled by dense search")
    vector_fetch_k: int = Field(20, ge=1, description="Candidates fetched before parent expansion")
    ensemble_weights: Annotated[list[float], NoDecode] = Field(
        default_factory=lambda: [0.4, 0.6],
        description="[bm25, vector] fusion weights",
    )

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------
    rerank_enabled: bool = Field(True, description="Enable Cohere cross-encoder reranking")
    rerank_model: str = Field("rerank-v3.5", description="Cohere rerank model")
    rerank_max_candidates: int = Field(40, ge=1, description="Cap on documents sent to the reranker")
    rerank_min_score: float = Field(
        0.50,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum reranker relevance for a passage to count as supporting evidence. "
            "Passages below it are dropped, and if none clear it the service declines to "
            "answer rather than citing unrelated documents."
        ),
    )

    # ------------------------------------------------------------------
    # Faithfulness Guard
    # ------------------------------------------------------------------
    enable_faithfulness_guard: bool = Field(
        True, description="Enable post-generation LLM entailment check to block hallucinations."
    )
    rag_persona: str = Field(
        "",
        description="Optional platform-wide instructions prepended for every tenant (Layer: deployment).",
    )
    prompt_pack_id: str = Field(
        "default",
        description="Prompt pack file id under src/core/prompts/{id}.json (version field inside the file).",
    )
    # ------------------------------------------------------------------
    # PostgreSQL (production document store + FTS keyword index)
    # When set, PostgresDocStore and PostgresBM25Retriever are used
    # instead of the local pickle files.
    # ------------------------------------------------------------------
    database_url: str | None = Field(
        None,
        description="SQLAlchemy DSN for PostgreSQL, e.g. postgresql+psycopg://user:pw@host/db",
    )

    # ------------------------------------------------------------------
    # Semantic cache
    # ------------------------------------------------------------------
    semantic_cache_enabled: bool = Field(True, description="Enable the Redis semantic cache")
    semantic_cache_threshold: float = Field(0.88, ge=0.0, le=1.0, description="Cosine similarity required for a hit")
    semantic_cache_ttl_seconds: int = Field(86400, ge=1, description="Cache entry lifetime in seconds")
    semantic_cache_max_candidates: int = Field(3, ge=1, description="Neighbours fetched per cache lookup")
    redis_url: str = Field("redis://localhost:6379/0", description="Redis connection URL")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    log_level: str = Field("INFO", description="Root log level")
    log_json: bool = Field(True, description="Emit structured JSON logs")

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("api_keys", "cors_origins", mode="before")
    @classmethod
    def _parse_string_list(cls, value: Any) -> list[str]:
        return _split_csv(value)

    @field_validator("ensemble_weights", mode="before")
    @classmethod
    def _parse_weights(cls, value: Any) -> list[float]:
        return _split_floats(value)

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: Any) -> str:
        return str(value).upper()

    @model_validator(mode="after")
    def _validate_embedding_dim(self) -> Settings:
        expected = EMBEDDING_DIMS.get(self.embedding_model)
        if expected is None:
            logger.warning(
                "Unknown embedding model '%s' — trusting EMBEDDING_DIM=%d. "
                "Add it to EMBEDDING_DIMS to enable the consistency check.",
                self.embedding_model,
                self.embedding_dim,
            )
        elif expected != self.embedding_dim:
            raise ValueError(
                f"EMBEDDING_DIM={self.embedding_dim} does not match "
                f"'{self.embedding_model}' which produces {expected}-dimensional vectors. "
                f"Set EMBEDDING_DIM={expected} (and re-ingest with --reset)."
            )
        return self

    @model_validator(mode="after")
    def _validate_ensemble_weights(self) -> Settings:
        weights = self.ensemble_weights
        if len(weights) != 2:
            raise ValueError(f"ENSEMBLE_WEIGHTS must contain exactly 2 values [bm25, vector], got {weights}")
        if any(weight < 0 for weight in weights):
            raise ValueError(f"ENSEMBLE_WEIGHTS cannot be negative: {weights}")
        if sum(weights) <= 0:
            raise ValueError(f"ENSEMBLE_WEIGHTS must sum to a positive value: {weights}")
        return self

    @model_validator(mode="after")
    def _validate_production_hardening(self) -> Settings:
        from src.config.production_guard import collect_production_errors

        errors = collect_production_errors(self)
        if errors:
            raise ValueError("Production configuration invalid:\n" + "\n".join(f"  - {item}" for item in errors))
        return self

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------
    @property
    def auth_enabled(self) -> bool:
        """Auth is on when Postgres multi-tenancy is enabled or legacy API_KEYS are set."""
        return bool(self.database_url) or bool(self.api_keys)

    @property
    def use_remote_qdrant(self) -> bool:
        return bool(self.qdrant_url)

    @property
    def bm25_weight(self) -> float:
        return self.ensemble_weights[0]

    @property
    def vector_weight(self) -> float:
        return self.ensemble_weights[1]

    @property
    def effective_cors_origins(self) -> list[str]:
        """Production CORS list, including https://SITE_DOMAIN when configured."""
        origins = list(self.cors_origins)
        if self.app_env.lower() == "production" and self.site_domain.strip():
            origin = f"https://{self.site_domain.strip()}"
            if origin not in origins:
                origins.append(origin)
        return origins

    @property
    def jwt_signing_key(self) -> str:
        """Key used to sign login tokens."""
        if self.jwt_secret:
            return self.jwt_secret
        if self.admin_api_key:
            import hashlib

            return hashlib.sha256(f"jwt:{self.admin_api_key}".encode()).hexdigest()
        return "insecure-dev-only-set-JWT_SECRET"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance, constructing it on first use."""
    return Settings()
