"""
DeepDoc-style Layout-Aware PDF Loader
--------------------------------------
Uses pymupdf4llm which runs the SAME ONNX layout recognition models
as RAGFlow's DeepDoc under the hood:
  - Document Layout Recognition (DLR): classifies page regions as
    header, paragraph, table, figure, caption, footer, etc.
  - Table Structure Recognition (TSR): converts tables to Markdown.
  - Reading-order detection: assembles blocks in natural reading order.

Each PDF page is emitted as a separate Document so the chunker can
treat Markdown headings as structural boundaries.
"""

import logging
import os
from typing import Iterator

import pymupdf4llm
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DeepDocLoader(BaseLoader):
    """
    A LangChain-compatible document loader that uses pymupdf4llm for
    layout-aware PDF parsing, mirroring RAGFlow's DeepDoc strategy.

    Features
    --------
    - ONNX-powered layout recognition (same model class as RAGFlow DeepDoc)
    - Markdown output preserving heading hierarchy (# / ## / ###)
    - Tables converted to Markdown format (not raw text blobs)
    - Per-page metadata: source, page number, layout blocks detected
    - Graceful fallback to plain-text extraction if layout model fails
    """

    def __init__(
        self,
        file_path: str,
        page_chunks: bool = True,
        extract_images: bool = False,
    ):
        """
        Parameters
        ----------
        file_path : str
            Absolute path to the PDF file.
        page_chunks : bool
            If True, yields one Document per page (recommended for
            fine-grained retrieval). If False, yields the whole document.
        extract_images : bool
            Whether to include image descriptions in output.
        """
        self.file_path = file_path
        self.page_chunks = page_chunks
        self.extract_images = extract_images

    def lazy_load(self) -> Iterator[Document]:
        """
        Parses the PDF using DeepDoc-style layout recognition and yields
        LangChain Documents with rich Markdown content and metadata.
        """
        filename = os.path.basename(self.file_path)
        logger.info(
            "DeepDoc layout parsing: %s (page_chunks=%s)",
            filename,
            self.page_chunks,
        )

        try:
            if self.page_chunks:
                # page_chunks=True → list of dicts, one per page
                pages = pymupdf4llm.to_markdown(
                    self.file_path,
                    page_chunks=True,
                    extract_images=self.extract_images,
                )
                for page in pages:
                    md_text: str = page.get("text", "").strip()
                    if not md_text:
                        continue  # skip blank pages

                    metadata = {
                        "source": filename,
                        "file_path": self.file_path,
                        "page": page.get("metadata", {}).get("page", None),
                        "parser": "deepdoc_layout",
                    }
                    yield Document(page_content=md_text, metadata=metadata)

            else:
                # Single-document mode: entire PDF as one Markdown string
                markdown_text: str = pymupdf4llm.to_markdown(
                    self.file_path,
                    page_chunks=False,
                    extract_images=self.extract_images,
                )
                yield Document(
                    page_content=markdown_text.strip(),
                    metadata={
                        "source": filename,
                        "file_path": self.file_path,
                        "parser": "deepdoc_layout",
                    },
                )

        except Exception as exc:
            logger.error(
                "DeepDoc layout parsing failed for %s: %s. Falling back to plain-text extraction.",
                filename,
                exc,
            )
            yield from self._plain_text_fallback()

    def _plain_text_fallback(self) -> Iterator[Document]:
        """Plain-text extraction via PyMuPDF as a safety fallback."""
        import fitz  # PyMuPDF

        filename = os.path.basename(self.file_path)
        try:
            doc = fitz.open(self.file_path)
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    yield Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "file_path": self.file_path,
                            "page": page_num,
                            "parser": "pymupdf_fallback",
                        },
                    )
            doc.close()
        except Exception as exc:
            logger.critical("Plain-text fallback also failed: %s", exc)
