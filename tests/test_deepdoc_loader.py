"""Tests for the layout-aware PDF loader.

The loader is the only place page metadata is produced, so a wrong key here makes
every citation in the system read "page None". That bug shipped once already.
"""

from __future__ import annotations

import pytest

from src.ingestion import deepdoc_loader
from src.ingestion.deepdoc_loader import DeepDocLoader


def _fake_pages(pages: list[dict]) -> callable:
    def fake_to_markdown(*args, **kwargs):
        del args, kwargs
        return pages

    return fake_to_markdown


def test_page_number_is_mapped_onto_page_metadata(monkeypatch):
    """pymupdf4llm exposes ``page_number``; the loader must not look for ``page``."""
    monkeypatch.setattr(
        deepdoc_loader.pymupdf4llm,
        "to_markdown",
        _fake_pages(
            [
                {"metadata": {"page_number": 1}, "text": "# Title\n\nbody one"},
                {"metadata": {"page_number": 2}, "text": "body two"},
            ]
        ),
    )

    documents = list(DeepDocLoader("corpus/a.pdf", page_chunks=True).lazy_load())

    assert [document.metadata["page"] for document in documents] == [1, 2]


def test_source_and_parser_metadata_are_attached(monkeypatch):
    monkeypatch.setattr(
        deepdoc_loader.pymupdf4llm,
        "to_markdown",
        _fake_pages([{"metadata": {"page_number": 7}, "text": "content"}]),
    )

    document = next(iter(DeepDocLoader("corpus/nested/paper.pdf", page_chunks=True).lazy_load()))

    assert document.metadata["source"] == "paper.pdf"
    assert document.metadata["parser"] == "deepdoc_layout"
    assert document.metadata["file_path"] == "corpus/nested/paper.pdf"


def test_blank_pages_are_skipped(monkeypatch):
    monkeypatch.setattr(
        deepdoc_loader.pymupdf4llm,
        "to_markdown",
        _fake_pages(
            [
                {"metadata": {"page_number": 1}, "text": "   "},
                {"metadata": {"page_number": 2}, "text": "real content"},
            ]
        ),
    )

    documents = list(DeepDocLoader("corpus/a.pdf", page_chunks=True).lazy_load())

    assert len(documents) == 1
    assert documents[0].metadata["page"] == 2


def test_missing_page_number_does_not_crash(monkeypatch):
    monkeypatch.setattr(
        deepdoc_loader.pymupdf4llm,
        "to_markdown",
        _fake_pages([{"metadata": {}, "text": "content"}]),
    )

    document = next(iter(DeepDocLoader("corpus/a.pdf", page_chunks=True).lazy_load()))

    assert document.metadata["page"] is None


def test_layout_failure_falls_back_to_plain_text(monkeypatch):
    """A parse failure must degrade to PyMuPDF text rather than dropping the file."""
    import fitz

    def boom(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("layout model unavailable")

    class _Page:
        def get_text(self, mode: str) -> str:
            del mode
            return "fallback text"

    class _Doc:
        def __iter__(self):
            return iter([_Page()])

        def close(self) -> None:
            return None

    monkeypatch.setattr(deepdoc_loader.pymupdf4llm, "to_markdown", boom)
    monkeypatch.setattr(fitz, "open", lambda path: _Doc())

    documents = list(DeepDocLoader("corpus/a.pdf", page_chunks=True).lazy_load())

    assert len(documents) == 1
    assert documents[0].page_content == "fallback text"
    assert documents[0].metadata["parser"] == "pymupdf_fallback"
    assert documents[0].metadata["page"] == 1


def test_single_document_mode_yields_one_document(monkeypatch):
    monkeypatch.setattr(
        deepdoc_loader.pymupdf4llm,
        "to_markdown",
        _fake_pages("# Whole paper\n\ncontent"),
    )

    documents = list(DeepDocLoader("corpus/a.pdf", page_chunks=False).lazy_load())

    assert len(documents) == 1
    assert documents[0].page_content.startswith("# Whole paper")
    assert documents[0].metadata["parser"] == "deepdoc_layout"


@pytest.mark.parametrize("page_number", [1, 12, 999])
def test_page_numbers_are_passed_through_unchanged(monkeypatch, page_number):
    monkeypatch.setattr(
        deepdoc_loader.pymupdf4llm,
        "to_markdown",
        _fake_pages([{"metadata": {"page_number": page_number}, "text": "x"}]),
    )

    document = next(iter(DeepDocLoader("corpus/a.pdf", page_chunks=True).lazy_load()))

    assert document.metadata["page"] == page_number
