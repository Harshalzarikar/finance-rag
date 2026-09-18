"""Tests for the persisted BM25 keyword retriever.

These guard the regression that made hybrid search unusable: the index used to be
pickled as a bare dict and handed to ``EnsembleRetriever`` as if it were a
retriever, and it stored no metadata so keyword hits could not be cited.
"""

from __future__ import annotations

import pickle

import pytest
from rank_bm25 import BM25Okapi

from src.retrieval.bm25 import (
    build_index,
    load_index,
    load_raw,
    save_index,
    tokenize,
)


def test_tokenize_is_case_insensitive_and_strips_punctuation():
    assert tokenize("Black-Scholes GARCH(1,1)!") == ["black", "scholes", "garch", "1", "1"]


def test_tokenize_is_identical_for_index_and_query_forms():
    assert tokenize("Volatility  Smile") == tokenize("volatility smile")


def test_build_index_returns_none_without_text():
    assert build_index([], []) is None


def test_build_index_rejects_mismatched_metadata():
    with pytest.raises(ValueError):
        build_index(["a", "b"], [{"source": "only-one.pdf"}])


def test_hits_carry_source_page_and_score():
    index = build_index(
        ["garch volatility modelling", "entirely unrelated content", "another garch paper"],
        [
            {"source": "a.pdf", "page": 1},
            {"source": "b.pdf", "page": 2},
            {"source": "c.pdf", "page": 3},
        ],
        k=2,
    )

    hits = index.invoke("garch volatility")

    assert hits
    assert hits[0].metadata["source"] == "a.pdf"
    assert hits[0].metadata["page"] == 1
    assert hits[0].metadata["retriever"] == "bm25"
    assert hits[0].metadata["bm25_score"] > 0


def test_query_without_lexical_overlap_returns_nothing():
    index = build_index(["alpha beta"], [{"source": "a.pdf"}], k=5)

    assert index.invoke("zzzz qqqq") == []


def test_k_limits_the_number_of_hits():
    index = build_index(
        [f"term{number} filler" for number in range(10)],
        [{"source": f"{number}.pdf"} for number in range(10)],
        k=3,
    )

    assert len(index.invoke("term1 term2 term3 term4 term5")) == 3


def test_round_trip_preserves_metadata(tmp_path):
    path = str(tmp_path / "index.pkl")
    save_index(build_index(["garch model volatility"], [{"source": "a.pdf", "page": 4}], k=1), path)

    loaded = load_index(path, k=1)

    assert loaded is not None
    hits = loaded.invoke("garch")
    assert hits[0].metadata["source"] == "a.pdf"
    assert hits[0].metadata["page"] == 4


def test_load_index_returns_none_when_missing(tmp_path):
    assert load_index(str(tmp_path / "absent.pkl"), k=5) is None


def test_load_index_tolerates_legacy_payload_without_metadata(tmp_path):
    """v1 pickles stored only {"bm25", "corpus"}; they must degrade, not crash."""
    path = str(tmp_path / "legacy.pkl")
    corpus = ["garch volatility"]
    with open(path, "wb") as handle:
        pickle.dump({"bm25": BM25Okapi([tokenize(text) for text in corpus]), "corpus": corpus}, handle)

    loaded = load_index(path, k=1)

    assert loaded is not None
    assert loaded.invoke("garch")[0].metadata.get("source") is None


def test_load_raw_rejects_unrecognised_payload(tmp_path):
    path = str(tmp_path / "bad.pkl")
    with open(path, "wb") as handle:
        pickle.dump(["not", "a", "dict"], handle)

    assert load_raw(path)["corpus"] == []


def test_load_raw_handles_corrupt_file(tmp_path):
    path = str(tmp_path / "corrupt.pkl")
    (tmp_path / "corrupt.pkl").write_bytes(b"definitely not a pickle")

    assert load_raw(path)["corpus"] == []
