"""
Layout-Aware Markdown Chunker
-------------------------------
Designed to work with the DeepDocLoader output, which produces structured
Markdown with heading hierarchy (#, ##, ###) from ONNX layout recognition.

RAGFlow reference: RAGFlow's "Naive" and "Paper" chunk templates split on
section boundaries detected by DLR, keeping semantic units together.

This chunker replicates that behavior:
  - Parent chunks  → one section per top-level heading (large context window)
  - Child chunks   → paragraph-sized splits for dense retrieval
"""

import re
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

# ---------------------------------------------------------------------------
# Section-boundary splitter (parent chunks)
# ---------------------------------------------------------------------------


class MarkdownSectionSplitter(TextSplitter):
    """
    Splits Markdown text on heading boundaries (produced by DeepDoc layout
    recognition). Keeps each section together as a parent chunk.

    Hierarchy:  # > ## > ### > ####
    Fallback:   if no headings found, falls back to paragraph splitting.
    """

    # Matches any Markdown heading line: #, ##, ###, ####
    _HEADING = re.compile(r"^#{1,4}\s+.+", re.MULTILINE)

    def __init__(self, max_chunk_size: int = 4000, **kwargs):
        super().__init__(**kwargs)
        self.max_chunk_size = max_chunk_size

    # ------------------------------------------------------------------
    def split_text(self, text: str) -> List[str]:
        sections = self._split_on_headings(text)
        chunks: List[str] = []

        for section in sections:
            section = section.strip()
            if not section:
                continue
            # If a single section exceeds max size, sub-split by paragraph
            if len(section) > self.max_chunk_size:
                chunks.extend(self._split_by_paragraph(section))
            else:
                chunks.append(section)

        return chunks if chunks else [text]

    # ------------------------------------------------------------------
    def _split_on_headings(self, text: str) -> List[str]:
        """Splits text at every Markdown heading boundary."""
        boundaries = [m.start() for m in self._HEADING.finditer(text)]
        if not boundaries:
            # No Markdown structure — fall back to paragraph splitting
            return self._split_by_paragraph(text)

        sections: List[str] = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            sections.append(text[start:end])
        return sections

    # ------------------------------------------------------------------
    @staticmethod
    def _split_by_paragraph(text: str) -> List[str]:
        """Paragraph-level fallback split (double newline boundary)."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs if paragraphs else [text]


# ---------------------------------------------------------------------------
# Paragraph-level splitter (child chunks for dense retrieval)
# ---------------------------------------------------------------------------


class DeepDocChildSplitter(TextSplitter):
    """
    Fine-grained child splitter for dense vector retrieval.

    Mirrors RAGFlow's strategy of breaking parent sections into small,
    semantically coherent child chunks (~512 tokens / ~700 chars) with
    overlap so that retrieved snippets have enough context.
    """

    # Split priority: sentence > paragraph > word
    _SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 80, **kwargs):
        super().__init__(**kwargs)
        self._splitter = RecursiveCharacterTextSplitter(
            separators=self._SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        return self._splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_splitters():
    """
    Returns (parent_splitter, child_splitter) for ParentDocumentRetriever.

    Parent: MarkdownSectionSplitter — section-level, heading-aware
    Child : DeepDocChildSplitter    — paragraph/sentence-level, overlapping

    Sizes come from configuration so ingestion and retrieval always agree.
    """
    from src.config.settings import get_settings

    settings = get_settings()
    parent_splitter = MarkdownSectionSplitter(max_chunk_size=settings.parent_chunk_size)
    child_splitter = DeepDocChildSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
    )
    return parent_splitter, child_splitter
