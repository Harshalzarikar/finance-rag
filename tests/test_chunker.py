"""Tests for the layout-aware chunkers."""

from __future__ import annotations

from langchain_core.documents import Document

from src.config.settings import get_settings
from src.ingestion.chunker import DeepDocChildSplitter, MarkdownSectionSplitter, get_splitters


def test_splits_on_heading_boundaries():
    splitter = MarkdownSectionSplitter(max_chunk_size=4000)
    text = "# Intro\nAlpha.\n\n## Method\nBeta.\n\n### Detail\nGamma."

    sections = splitter.split_text(text)

    assert len(sections) == 3
    assert sections[0].startswith("# Intro")
    assert sections[1].startswith("## Method")
    assert sections[2].startswith("### Detail")


def test_oversized_section_is_split_by_paragraph():
    splitter = MarkdownSectionSplitter(max_chunk_size=60)
    text = "# Big\n\n" + "\n\n".join(f"Paragraph {index} " + "x" * 40 for index in range(4))

    sections = splitter.split_text(text)

    assert len(sections) > 1
    assert sections[0] == "# Big"


def test_text_without_headings_falls_back_to_paragraphs():
    splitter = MarkdownSectionSplitter(max_chunk_size=4000)

    assert splitter.split_text("Para one.\n\nPara two.") == ["Para one.", "Para two."]


def test_child_splitter_respects_chunk_size():
    splitter = DeepDocChildSplitter(chunk_size=100, chunk_overlap=20)

    chunks = splitter.split_text(" ".join(f"word{index}" for index in range(200)))

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_child_splitter_preserves_citation_metadata():
    """Child chunks must keep source/page, or keyword citations become unresolvable."""
    splitter = DeepDocChildSplitter(chunk_size=200, chunk_overlap=20)
    documents = [Document(page_content="word " * 60, metadata={"source": "a.pdf", "page": 7})]

    children = splitter.split_documents(documents)

    assert len(children) > 1
    assert all(child.metadata["source"] == "a.pdf" for child in children)
    assert all(child.metadata["page"] == 7 for child in children)


def test_get_splitters_reads_configured_sizes(monkeypatch):
    monkeypatch.setenv("PARENT_CHUNK_SIZE", "1234")
    monkeypatch.setenv("CHILD_CHUNK_SIZE", "321")
    monkeypatch.setenv("CHILD_CHUNK_OVERLAP", "11")
    get_settings.cache_clear()

    parent, child = get_splitters()

    assert parent.max_chunk_size == 1234
    assert child._splitter._chunk_size == 321
    assert child._splitter._chunk_overlap == 11
