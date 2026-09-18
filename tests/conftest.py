"""Shared test fixtures.

Configuration is injected through environment variables before any ``src`` module
imports settings, and every cached factory is cleared around each test so tests
cannot leak configuration or heavyweight clients into one another.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

# Environment variables take precedence over .env in pydantic-settings, so these
# also neutralise a developer's real credentials during tests.
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("COHERE_API_KEY", "test-cohere-key")
os.environ.setdefault("API_KEYS", "test-api-key")
os.environ.setdefault("QDRANT_URL", "")
os.environ.setdefault("SEMANTIC_CACHE_ENABLED", "false")
os.environ.setdefault("LOG_JSON", "false")

TEST_API_KEY = "test-api-key"


def clear_factory_caches() -> None:
    """Reset every cached factory so the next call re-reads configuration."""
    from src.api.ratelimit import get_rate_limiter
    from src.cache.semantic_cache import get_semantic_cache
    from src.config.settings import get_settings
    from src.core.rag_pipeline import get_rag_pipeline
    from src.llm.generator import get_llm
    from src.llm.reranker import get_reranker
    from src.retrieval.hybrid_search import get_hybrid_search
    from src.retrieval.vector_store import (
        get_docstore,
        get_embeddings,
        get_qdrant_client,
        get_retriever,
        get_vectorstore,
    )

    for factory in (
        get_settings,
        get_rag_pipeline,
        get_llm,
        get_reranker,
        get_hybrid_search,
        get_docstore,
        get_embeddings,
        get_qdrant_client,
        get_retriever,
        get_vectorstore,
        get_semantic_cache,
        get_rate_limiter,
    ):
        factory.cache_clear()


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Point every filesystem-backed setting at a throwaway directory."""
    monkeypatch.setenv("QDRANT_DB_DIR", str(tmp_path / "qdrant_db"))
    monkeypatch.setenv("DOC_STORE_DIR", str(tmp_path / "doc_store"))
    monkeypatch.setenv("RAW_PDFS_DIR", str(tmp_path / "pdfs"))
    monkeypatch.setenv("BM25_INDEX_FILE", str(tmp_path / "bm25_index.pkl"))
    monkeypatch.setenv("INGESTION_MANIFEST_FILE", str(tmp_path / "manifest.json"))
    (tmp_path / "pdfs").mkdir(parents=True, exist_ok=True)

    clear_factory_caches()
    yield
    clear_factory_caches()


def make_document(content: str, source: str = "paper.pdf", page: int = 1, **metadata: Any):
    """Build a LangChain Document with citation metadata."""
    from langchain_core.documents import Document

    return Document(page_content=content, metadata={"source": source, "page": page, **metadata})


class FakeRetriever:
    """Stand-in for HybridSearchService."""

    def __init__(self, documents: list[Any], hybrid_enabled: bool = True, chunks: int = 5) -> None:
        self.documents = documents
        self.hybrid_enabled = hybrid_enabled
        self.bm25_retriever = type("BM25Stub", (), {"corpus": ["chunk"] * chunks})()
        self.queries: list[str] = []

    def search(self, query: str) -> list[Any]:
        self.queries.append(query)
        return list(self.documents)


class FakeReranker:
    """Stand-in for CohereReranker that preserves irrelevant metadata untouched."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.model = "rerank-v3.5"

    def rerank(self, query: str, documents: list[Any], top_n: int = 5) -> list[Any]:
        del query
        from langchain_core.documents import Document

        return [
            Document(page_content=doc.page_content, metadata={**doc.metadata, "relevance_score": 0.5})
            for doc in documents[:top_n]
        ]


class FakeGenerator:
    """Stand-in for a chat model that reports a model name for health checks."""

    model_name = "fake-model"


class FakeChain:
    """Mimics ``prompt | llm`` for both invoke and stream."""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def invoke(self, payload: dict[str, Any]) -> Any:
        del payload
        return type("Message", (), {"content": self.answer})()

    def stream(self, payload: dict[str, Any]):
        del payload
        for index, token in enumerate(self.answer.split(" ")):
            yield type("Message", (), {"content": token if index == 0 else f" {token}"})()


class FakeCache:
    """In-memory semantic cache with the same surface as SemanticCache."""

    def __init__(self, hit: Any = None) -> None:
        self.hit = hit
        self.stored: list[tuple[str, str, list[dict[str, Any]]]] = []

    def get(self, query: str):
        del query
        return self.hit

    def set(self, query: str, answer: str, sources: list[dict[str, Any]]) -> None:
        self.stored.append((query, answer, sources))


class FakePipeline:
    """Stand-in for RAGPipeline used by the API tests."""

    def __init__(self, answer: str = "A grounded answer.") -> None:
        self.retriever = FakeRetriever([])
        self.reranker = FakeReranker()
        self.generator = FakeGenerator()
        self.answer = answer
        self.calls: list[tuple[str, int]] = []

    def run(self, query: str, top_k_rerank: int = 5) -> dict[str, Any]:
        self.calls.append((query, top_k_rerank))
        return {
            "answer": self.answer,
            "sources": [{"source": "paper.pdf", "page": 3, "score": 0.91, "snippet": "excerpt"}],
            "cached": False,
            "cache_similarity": None,
        }

    def stream(self, query: str, top_k_rerank: int = 5):
        del query, top_k_rerank
        yield {
            "type": "sources",
            "sources": [{"source": "paper.pdf", "page": 3, "score": 0.91, "snippet": "excerpt"}],
            "cached": False,
        }
        yield {"type": "token", "value": "A "}
        yield {"type": "token", "value": "grounded answer."}
        yield {"type": "done", "cached": False}


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Headers carrying a valid API key for the configured API_KEYS value."""
    return {"X-API-Key": os.environ.get("API_KEYS", TEST_API_KEY).split(",")[0]}


@pytest.fixture
def client(fake_pipeline: FakePipeline):
    """TestClient with the real pipeline replaced by a stub.

    The lifespan is deliberately not run: warming the real pipeline would load the
    embedding model on every test.
    """
    from fastapi.testclient import TestClient

    import src.api.main as main
    from src.api.deps import get_pipeline

    main.app.dependency_overrides[get_pipeline] = lambda: fake_pipeline
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()
