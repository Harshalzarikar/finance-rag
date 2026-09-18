"""Tests for cross-encoder reranking, including its failure modes."""

from __future__ import annotations

from langchain_core.documents import Document

from src.llm.reranker import CohereReranker


class _Result:
    def __init__(self, index: int, score: float) -> None:
        self.index = index
        self.relevance_score = score


class _Response:
    def __init__(self, results: list[_Result]) -> None:
        self.results = results


class _FakeClient:
    def __init__(self, results: list[_Result] | None = None, error: Exception | None = None) -> None:
        self._results = results or []
        self._error = error

    def rerank(self, **kwargs):
        if self._error is not None:
            raise self._error
        top_n = kwargs.get("top_n", len(self._results))
        return _Response(self._results[:top_n])


def _documents() -> list[Document]:
    return [
        Document(page_content="first", metadata={"source": "a.pdf"}),
        Document(page_content="second", metadata={"source": "b.pdf"}),
    ]


def test_disabled_without_an_api_key():
    assert CohereReranker(api_key="", model="rerank-v3.5").enabled is False


def test_disabled_by_configuration():
    assert CohereReranker(api_key="key", model="rerank-v3.5", enabled=False).enabled is False


def test_falls_back_to_input_order_when_disabled():
    reranker = CohereReranker(api_key="", model="rerank-v3.5")

    result = reranker.rerank("query", _documents(), top_n=1)

    assert [document.page_content for document in result] == ["first"]


def test_falls_back_when_the_api_raises():
    reranker = CohereReranker(api_key="key", model="rerank-v3.5")
    reranker.client = _FakeClient(error=RuntimeError("cohere is down"))

    result = reranker.rerank("query", _documents(), top_n=2)

    assert [document.page_content for document in result] == ["first", "second"]


def test_reorders_by_score_and_attaches_the_score():
    reranker = CohereReranker(api_key="key", model="rerank-v3.5")
    reranker.client = _FakeClient(results=[_Result(1, 0.9), _Result(0, 0.1)])

    result = reranker.rerank("query", _documents(), top_n=2)

    assert [document.page_content for document in result] == ["second", "first"]
    assert result[0].metadata["relevance_score"] == 0.9


def test_inputs_are_not_mutated():
    """Retrieved Documents are shared, so reranking must not edit them in place."""
    reranker = CohereReranker(api_key="key", model="rerank-v3.5")
    reranker.client = _FakeClient(results=[_Result(1, 0.9)])
    documents = _documents()

    reranker.rerank("query", documents, top_n=1)

    assert "relevance_score" not in documents[0].metadata
    assert "relevance_score" not in documents[1].metadata


def test_empty_input_returns_empty():
    reranker = CohereReranker(api_key="key", model="rerank-v3.5")
    reranker.client = _FakeClient()

    assert reranker.rerank("query", [], top_n=5) == []


def test_top_n_truncates_the_candidates():
    reranker = CohereReranker(api_key="key", model="rerank-v3.5")
    reranker.client = _FakeClient(results=[_Result(0, 0.9), _Result(1, 0.8)])

    assert len(reranker.rerank("query", _documents(), top_n=1)) == 1
