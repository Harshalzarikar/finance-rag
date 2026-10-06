"""Tests for the RAG pipeline's assembly, deduplication, and caching behaviour.

The generator is supplied through the ``_chain`` seam so no model is ever called.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from src.core.rag_pipeline import RAGPipeline


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Chain:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def invoke(self, payload: dict[str, Any]) -> _Message:
        del payload
        return _Message(self.answer)

    def stream(self, payload: dict[str, Any]):
        del payload
        for index, token in enumerate(self.answer.split(" ")):
            yield _Message(token if index == 0 else f" {token}")


class _Retriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.queries: list[str] = []

    def search(self, query: str) -> list[Document]:
        self.queries.append(query)
        return list(self.documents)


class _Reranker:
    def __init__(self) -> None:
        self.seen: list[Document] = []

    def rerank(self, query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
        del query
        self.seen = list(documents)
        return [
            Document(page_content=doc.page_content, metadata={**doc.metadata, "relevance_score": 0.7})
            for doc in documents[:top_n]
        ]


class _Cache:
    def __init__(self, hit: Any = None) -> None:
        self.hit = hit
        self.stored: list[tuple[str, str, list[dict[str, Any]]]] = []

    def get(self, query: str) -> Any:
        del query
        return self.hit

    def set(self, query: str, answer: str, sources: list[dict[str, Any]]) -> None:
        self.stored.append((query, answer, sources))


class _Store:
    def __init__(self, documents: dict[str, Document]) -> None:
        self.documents = documents

    def mget(self, keys: list[str]) -> list[Document | None]:
        return [self.documents.get(key) for key in keys]


class _Hit:
    def __init__(self, answer: str, sources: list[dict[str, Any]]) -> None:
        self.answer = answer
        self.sources = sources
        self.similarity = 0.93


class _DummyLLM:
    def with_structured_output(self, *args, **kwargs):
        from langchain_core.runnables import RunnableLambda

        from src.core.faithfulness_guard import FaithfulnessResult

        return RunnableLambda(lambda x: FaithfulnessResult(is_faithful=True))


def _build(
    documents: list[Document] | None = None,
    cache: _Cache | None = None,
    store: _Store | None = None,
    reranker: _Reranker | None = None,
    answer: str = "A grounded answer.",
) -> RAGPipeline:
    pipeline = RAGPipeline(
        retriever=_Retriever(documents or []),
        reranker=reranker or _Reranker(),
        generator=_DummyLLM(),
        cache=cache or _Cache(),
        docstore=store,
    )
    pipeline._chain = lambda _query, _documents: _Chain(answer)  # type: ignore[method-assign]
    return pipeline


def _document(content: str, source: str = "paper.pdf", page: int = 1) -> Document:
    return Document(page_content=content, metadata={"source": source, "page": page})


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_duplicate_chunks_are_collapsed():
    pipeline = _build()
    duplicated = [_document("same text"), _document("same text"), _document("other text")]

    unique = pipeline._deduplicate(duplicated)

    assert [document.page_content for document in unique] == ["same text", "other text"]


def test_same_text_from_different_documents_is_kept():
    pipeline = _build()
    documents = [_document("shared boilerplate", source="a.pdf"), _document("shared boilerplate", source="b.pdf")]

    assert len(pipeline._deduplicate(documents)) == 2


# ---------------------------------------------------------------------------
# Parent resolution
# ---------------------------------------------------------------------------


def test_keyword_hits_are_expanded_to_their_parent_section():
    parent = Document(page_content="full section text", metadata={"source": "a.pdf", "page": 2})
    pipeline = _build(store=_Store({"id-1": parent}))
    child = Document(
        page_content="small keyword chunk",
        metadata={"source": "a.pdf", "page": 2, "doc_id": "id-1", "bm25_score": 1.5},
    )

    resolved = pipeline._resolve_parents([child])

    assert resolved[0].page_content == "full section text"
    assert resolved[0].metadata["bm25_score"] == 1.5
    assert resolved[0].metadata["source"] == "a.pdf"


def test_dense_hits_pass_through_unchanged():
    pipeline = _build(store=_Store({}))
    document = _document("parent returned by the vector store")

    resolved = pipeline._resolve_parents([document])

    assert resolved[0].page_content == "parent returned by the vector store"


def test_resolution_is_skipped_without_a_docstore():
    pipeline = _build()
    child = Document(page_content="chunk", metadata={"source": "a.pdf", "doc_id": "id-1"})

    assert pipeline._resolve_parents([child])[0].page_content == "chunk"


# ---------------------------------------------------------------------------
# Context and citations
# ---------------------------------------------------------------------------


def test_context_labels_every_passage_with_source_and_page():
    context = RAGPipeline._build_context([_document("body", source="a.pdf", page=4)])

    assert "[Source: a.pdf, page 4]" in context
    assert "body" in context


def test_context_handles_no_documents():
    assert "no relevant context" in RAGPipeline._build_context([])


def test_sources_expose_source_page_and_score():
    document = Document(page_content="body", metadata={"source": "a.pdf", "page": 4, "relevance_score": 0.42})

    sources = RAGPipeline._build_sources([document])

    assert sources[0]["source"] == "a.pdf"
    assert sources[0]["page"] == 4
    assert sources[0]["score"] == 0.42
    assert sources[0]["snippet"].startswith("body")


def test_unknown_source_is_reported_explicitly():
    document = Document(page_content="body", metadata={})

    assert RAGPipeline._build_sources([document])[0]["source"] == "Unknown"


# ---------------------------------------------------------------------------
# Cache integration
# ---------------------------------------------------------------------------


def test_cache_hit_short_circuits_retrieval():
    cache = _Cache(hit=_Hit("cached answer", [{"source": "a.pdf", "page": 1, "score": None, "snippet": "x"}]))
    retriever = _Retriever([_document("fresh")])
    pipeline = _build(cache=cache)
    pipeline.retriever = retriever

    result = pipeline.run("explain garch")

    assert result["answer"] == "cached answer"
    assert result["cached"] is True
    assert result["cache_similarity"] == pytest.approx(0.93)
    assert retriever.queries == []


def test_comprehensive_query_bypasses_semantic_cache():
    cache = _Cache(
        hit=_Hit("short cached answer", [{"source": "resume.pdf", "page": 1, "score": None, "snippet": "x"}])
    )
    retriever = _Retriever([_document("Experience section", source="resume.pdf", page=2)])
    pipeline = _build(cache=cache, documents=[_document("header", source="resume.pdf")])
    pipeline.retriever = retriever

    result = pipeline.run("tell me all details about harshal zarikar")

    assert result["cached"] is False
    assert retriever.queries == ["tell me all details about harshal zarikar"]
    assert cache.stored == []


def test_cache_miss_runs_the_pipeline_and_populates_the_cache():
    cache = _Cache()
    pipeline = _build(documents=[_document("body", source="a.pdf", page=4)], cache=cache)

    result = pipeline.run("explain garch")

    assert result["answer"] == "A grounded answer."
    assert result["cached"] is False
    assert cache.stored[0][0].endswith("explain garch")
    assert cache.stored[0][1] == "A grounded answer."
    assert cache.stored[0][2][0]["source"] == "a.pdf"


def test_candidates_are_capped_before_reranking(monkeypatch):
    monkeypatch.setenv("RERANK_MAX_CANDIDATES", "2")
    from src.config.settings import get_settings

    get_settings.cache_clear()

    reranker = _Reranker()
    pipeline = _build(
        documents=[_document(f"chunk {index}", source=f"{index}.pdf") for index in range(5)],
        reranker=reranker,
    )

    pipeline.run("explain garch")

    assert len(reranker.seen) == 2


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_emits_sources_before_tokens():
    pipeline = _build(documents=[_document("body", source="a.pdf", page=4)], answer="grounded answer here")

    events = list(pipeline.stream("explain garch"))

    assert events[0]["type"] == "sources"
    assert events[0]["sources"][0]["source"] == "a.pdf"
    assert events[-1]["type"] == "done"
    assert events[-1]["cached"] is False
    assert events[-1]["faithfulness_passed"] is True
    assert events[-1].get("prompt_pack_version", "").startswith("default@")
    joined = "".join(event["value"] for event in events if event["type"] == "token")
    assert joined == "grounded answer here"


def test_stream_replays_a_cache_hit_as_one_token():
    cache = _Cache(hit=_Hit("cached answer", [{"source": "a.pdf", "page": 1, "score": None, "snippet": "x"}]))
    pipeline = _build(cache=cache)

    events = list(pipeline.stream("explain garch"))

    assert events[0]["cached"] is True
    assert [event["value"] for event in events if event["type"] == "token"] == ["cached answer"]
    assert events[-1]["type"] == "done"
    assert events[-1]["cached"] is True
    assert events[-1]["faithfulness_passed"] is True
    assert events[-1].get("prompt_pack_version", "").startswith("default@")


def test_stream_stores_the_generated_answer():
    cache = _Cache()
    pipeline = _build(documents=[_document("body")], cache=cache, answer="streamed answer")

    list(pipeline.stream("explain garch"))

    assert cache.stored[0][0].endswith("explain garch")
    assert cache.stored[0][1] == "streamed answer"


# ---------------------------------------------------------------------------
# Relevance floor
# ---------------------------------------------------------------------------


class _ScoredReranker:
    """Assigns a fixed relevance score to each document, in order."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def rerank(self, query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
        del query
        return [
            Document(page_content=doc.page_content, metadata={**doc.metadata, "relevance_score": score})
            for doc, score in zip(documents[:top_n], self.scores, strict=False)
        ]


class _UnscoredReranker:
    """Mimics a disabled or failing reranker: returns documents with no score."""

    def rerank(self, query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
        del query
        return list(documents[:top_n])


def _corpus(count: int) -> list[Document]:
    return [_document(f"chunk {index}", source=f"{index}.pdf") for index in range(count)]


def test_passages_below_the_relevance_floor_are_not_cited():
    pipeline = _build(documents=_corpus(3), reranker=_ScoredReranker([0.91, 0.64, 0.05]))

    result = pipeline.run("explain garch")

    assert [source["score"] for source in result["sources"]] == [0.91, 0.64]


def test_unanswerable_query_declines_without_citations():
    """Regression: a 'no answer' reply used to still list the near-misses as citations."""
    pipeline = _build(documents=_corpus(3), reranker=_ScoredReranker([0.07, 0.05, 0.04]))

    result = pipeline.run("what is the heston model?")

    assert result["sources"] == []
    assert "No passage" in result["answer"]
    assert result["cached"] is False


def test_declining_skips_the_generator():
    pipeline = _build(documents=_corpus(2), reranker=_ScoredReranker([0.01, 0.01]))

    def explode(_query: Any, _documents: Any) -> None:
        raise AssertionError("the generator must not run when nothing is relevant")

    pipeline._chain = explode  # type: ignore[method-assign]

    assert pipeline.run("anything")["sources"] == []


def test_a_declined_answer_is_not_cached():
    """Caching a non-answer would outlive a re-index that makes it answerable."""
    cache = _Cache()
    pipeline = _build(documents=_corpus(2), cache=cache, reranker=_ScoredReranker([0.02, 0.01]))

    pipeline.run("anything")

    assert cache.stored == []


def test_missing_scores_disable_the_floor():
    """With no reranker there is no score to judge by, so nothing is dropped."""
    pipeline = _build(documents=_corpus(3), reranker=_UnscoredReranker())

    assert len(pipeline.run("explain garch")["sources"]) == 3


def test_the_floor_is_configurable(monkeypatch):
    from src.config.settings import get_settings

    monkeypatch.setenv("RERANK_MIN_SCORE", "0.95")
    get_settings.cache_clear()

    pipeline = _build(documents=_corpus(2), reranker=_ScoredReranker([0.90, 0.20]))

    assert pipeline.run("anything")["sources"] == []


def test_entity_query_keeps_name_matched_passages_below_floor(monkeypatch):
    """Person-name queries often score low on Cohere rerank; keep filename/body matches."""
    from src.config.settings import get_settings

    monkeypatch.setenv("RERANK_MIN_SCORE", "0.50")
    get_settings.cache_clear()

    resume = Document(
        page_content="Experience with ML systems.",
        metadata={"source": "Harshal_Zarikar_Resume.pdf"},
    )
    noise = Document(page_content="Long finance paper text.", metadata={"source": "paper.pdf"})
    pipeline = _build(
        documents=[resume, noise],
        reranker=_ScoredReranker([0.02, 0.03]),
        answer="Detailed profile.",
    )

    result = pipeline.run("harshal zarikar information in detail")

    assert result["sources"]
    assert "zarikar" in result["sources"][0]["source"].lower()
    assert "No passage" not in result["answer"]


def test_stream_declines_without_citations():
    pipeline = _build(documents=_corpus(2), reranker=_ScoredReranker([0.03, 0.02]))

    events = list(pipeline.stream("anything"))

    assert events[0]["sources"] == []
    tokens = "".join(event["value"] for event in events if event["type"] == "token")
    assert "No passage" in tokens
