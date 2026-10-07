"""Offline ingestion pipeline.

Stages
------
1. ``DeepDocLoader``           ONNX layout recognition -> Markdown per page
2. ``MarkdownSectionSplitter`` parent chunks on heading boundaries
3. ``DeepDocChildSplitter``    child chunks for dense retrieval
4. Qdrant                      child-chunk embeddings
5. ``PickleFileStore``         parent documents
6. BM25                        keyword index over the very same child chunks

Both retrieval indexes are built from one split, so a keyword hit and a dense hit
refer to the same unit of text and carry identical citation metadata.

Usage
-----
    python scripts/ingest.py                            # every PDF in real_pdfs/
    python scripts/ingest.py --pdf real_pdfs/foo.pdf    # one file
    python scripts/ingest.py --reset --limit 5          # wipe, then ingest the first 5
    python scripts/ingest.py --offset 100 --limit 50    # resume a batch
    python scripts/ingest.py --concurrency 4            # parallel PDF parsing
    python scripts/ingest.py --verify                   # manifest vs Qdrant drift check
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import shutil
import stat
import sys
import time
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from langchain_core.documents import Document  # noqa: E402

from src.config.settings import get_settings  # noqa: E402
from src.ingestion.deepdoc_loader import DeepDocLoader  # noqa: E402
from src.observability.logging import configure_logging  # noqa: E402
from src.retrieval import bm25  # noqa: E402
from src.retrieval.vector_store import (  # noqa: E402
    count_points,
    count_points_for_tenant,
    delete_source,
    ensure_collection,
    get_qdrant_client,
    get_retriever,
)

logger = logging.getLogger("ingest")

MANIFEST_VERSION = 2


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def source_key(pdf_path: str) -> str:
    """Portable manifest key: path relative to the corpus root, POSIX separators.

    Absolute paths would make the manifest host-specific and silently defeat
    idempotency inside a container.
    """
    root = os.path.abspath(get_settings().raw_pdfs_dir)
    absolute = os.path.abspath(pdf_path)
    try:
        relative = os.path.relpath(absolute, root)
    except ValueError:
        # Different drive on Windows — fall back to the basename.
        relative = os.path.basename(absolute)
    return relative.replace("\\", "/")


def manifest_target() -> dict[str, Any]:
    """Identify what a manifest describes, so a config change invalidates it.

    Skipping a file is only valid if its vectors actually exist in the collection
    this run targets. Pointing at a different collection, or changing the embedding
    model, makes every recorded entry meaningless — so the target is recorded and
    checked, and a mismatch discards the manifest rather than silently skipping
    work that was never done.
    """
    settings = get_settings()
    return {
        "collection": settings.qdrant_collection_name,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
    }


def load_manifest(path: str) -> dict[str, dict[str, Any]]:
    """Read the manifest, discarding it when it describes a different target."""
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read ingestion manifest (%s) — starting fresh.", exc)
        return {}

    if not isinstance(payload, dict):
        logger.warning("Ingestion manifest has an unexpected shape — starting fresh.")
        return {}

    wrapped = isinstance(payload.get("files"), dict)
    files = payload["files"] if wrapped else payload
    if not isinstance(files, dict):
        logger.warning("Ingestion manifest has an unexpected shape — starting fresh.")
        return {}

    recorded_target = payload.get("target") if wrapped else None
    current_target = manifest_target()
    if recorded_target != current_target:
        logger.warning(
            "Ignoring %d manifest entries: they describe %s, but this run targets %s. Every file will be re-indexed.",
            len(files),
            recorded_target or "an older manifest with no recorded target",
            current_target,
        )
        return {}

    return files


def save_manifest(path: str, files: dict[str, dict[str, Any]]) -> None:
    """Persist progress atomically so an interrupted run can resume."""
    temporary_path = f"{path}.tmp"
    payload = {"version": MANIFEST_VERSION, "target": manifest_target(), "files": files}
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


# ---------------------------------------------------------------------------
# Parsing and chunking
# ---------------------------------------------------------------------------


def file_hash(path: str) -> str:
    """Content hash used to detect changed files between ingestion runs."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_pdf(pdf_path: str) -> list[Document]:
    """Layout-aware parse of a single PDF into one Document per page."""
    return list(DeepDocLoader(pdf_path, page_chunks=True).lazy_load())


def chunk_for_bm25(documents: Sequence[Document], parent_splitter: Any, child_splitter: Any) -> list[Document]:
    """Reproduce the child chunks that ``ParentDocumentRetriever`` embeds.

    ``split_documents`` (rather than ``split_text``) is what carries ``source``
    and ``page`` onto every child, which is what makes keyword citations resolvable.
    """
    parents = parent_splitter.split_documents(list(documents))
    return child_splitter.split_documents(parents)


def index_documents(
    pdf_path: str, documents: Sequence[Document], retriever: Any, tenant_id: str = "default"
) -> list[Document]:
    """Index parsed pages, returning the child chunks to add to the keyword index.

    Any vectors previously stored for this file are dropped first, so re-ingesting
    a changed PDF replaces its content instead of duplicating it.
    """
    delete_source(os.path.basename(pdf_path))
    for document in documents:
        document.metadata["tenant_id"] = tenant_id
    retriever.add_documents(list(documents), ids=None)
    return chunk_for_bm25(documents, retriever.parent_splitter, retriever.child_splitter)


def _iter_parsed(
    pdf_paths: Sequence[str], concurrency: int
) -> Iterator[tuple[str, list[Document] | None, Exception | None]]:
    """Parse PDFs concurrently, yielding results in submission order.

    Layout recognition is CPU-bound and dominates runtime, so it is parallelised
    while indexing stays on the calling thread. In-flight work is bounded so
    memory stays flat regardless of corpus size.
    """
    if concurrency <= 1:
        for path in pdf_paths:
            try:
                yield path, parse_pdf(path), None
            except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the run
                yield path, None, exc
        return

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending: deque[tuple[str, Future[list[Document]]]] = deque()
        paths = iter(pdf_paths)

        for _ in range(concurrency * 2):
            path = next(paths, None)
            if path is None:
                break
            pending.append((path, pool.submit(parse_pdf, path)))

        while pending:
            path, future = pending.popleft()
            upcoming = next(paths, None)
            if upcoming is not None:
                pending.append((upcoming, pool.submit(parse_pdf, upcoming)))

            try:
                yield path, future.result(), None
            except Exception as exc:  # noqa: BLE001
                yield path, None, exc


# ---------------------------------------------------------------------------
# Store maintenance
# ---------------------------------------------------------------------------


def _rmtree_onerror(func: Any, path: str, exc_info: Any) -> None:
    """Retry handler for ``shutil.rmtree`` on Windows file locks."""
    del exc_info
    try:
        os.chmod(path, stat.S_IWRITE)
        time.sleep(0.3)
        func(path)
    except OSError:
        logger.error(
            "Cannot delete '%s' — it is locked by another process. Stop the API server and re-run.",
            path,
        )


def reset_stores() -> None:
    """Drop the collection, the document store, the BM25 index, and the manifest."""
    settings = get_settings()

    client = get_qdrant_client()
    if client.collection_exists(settings.qdrant_collection_name):
        client.delete_collection(settings.qdrant_collection_name)
        logger.info("Deleted Qdrant collection '%s'.", settings.qdrant_collection_name)

    if settings.database_url:
        from src.db.schema import ChildChunkFTS, ParentDocument, get_session_factory

        with get_session_factory(settings.database_url)() as session:
            session.query(ChildChunkFTS).delete()
            session.query(ParentDocument).delete()
            session.commit()
        logger.info("Cleared Postgres parent_documents and child_chunks_fts.")

    if os.path.isdir(settings.doc_store_dir):
        shutil.rmtree(settings.doc_store_dir, onerror=_rmtree_onerror)
        logger.info("Wiped %s", settings.doc_store_dir)

    for path in (settings.bm25_index_file, settings.ingestion_manifest_file):
        if os.path.exists(path):
            os.remove(path)
            logger.info("Wiped %s", path)


def rebuild_bm25(
    existing_corpus: list[str],
    existing_metadatas: list[dict[str, Any]],
    replaced_sources: set[str],
    new_chunks: list[Document],
) -> None:
    """Rebuild the keyword index, replacing chunks belonging to re-ingested files.

    Reusing the persisted corpus avoids re-parsing the whole archive on every run,
    while dropping ``replaced_sources`` entries keeps re-ingestion idempotent
    instead of letting stale chunks accumulate.
    """
    settings = get_settings()

    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for text, metadata in zip(existing_corpus, existing_metadatas, strict=False):
        if metadata.get("source") in replaced_sources:
            continue
        texts.append(text)
        metadatas.append(metadata)

    kept = len(texts)
    texts.extend(chunk.page_content for chunk in new_chunks)
    metadatas.extend(dict(chunk.metadata) for chunk in new_chunks)

    if not texts:
        logger.warning("No chunks available — skipping BM25 rebuild.")
        return

    index = bm25.build_index(texts, metadatas, k=settings.bm25_k)
    if index is not None:
        logger.info("BM25 index: %d retained + %d new chunks.", kept, len(new_chunks))
        bm25.save_index(index, settings.bm25_index_file)


def write_postgres_fts(
    database_url: str, new_chunks: Sequence[Document], replaced_sources: set[str], tenant_id: str = "default"
) -> None:
    """Write child chunks to the Postgres FTS index (production keyword index).

    Mirrors the Celery worker's ``_pg_upsert``: stale rows for re-ingested sources
    are deleted first so re-ingestion stays idempotent, then the fresh chunks are
    inserted. Used when ``DATABASE_URL`` is set, because the runtime then reads
    keyword hits from ``child_chunks_fts`` (``PostgresBM25Retriever``) rather than
    the local ``bm25_index.pkl``.
    """
    from src.db.pg_bm25 import delete_chunks_by_source, insert_chunks

    for source in replaced_sources:
        delete_chunks_by_source(database_url, source, tenant_id=tenant_id)
    if new_chunks:
        insert_chunks(database_url, new_chunks, tenant_id=tenant_id)
    logger.info(
        "Postgres FTS updated: %d chunks across %d replaced source(s).",
        len(new_chunks),
        len(replaced_sources),
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def run_ingestion(args: argparse.Namespace) -> int:
    """Ingest the selected PDFs, then rebuild the keyword index."""
    settings = get_settings()

    if args.reset:
        logger.warning("--reset: wiping the vector store, document store, and keyword index.")
        reset_stores()

    pdf_paths = _select_pdfs(args)
    if not pdf_paths:
        logger.error("No PDF files found in %s.", os.path.abspath(settings.raw_pdfs_dir))
        return 1

    logger.info("Found %d PDF(s) to ingest (concurrency=%d).", len(pdf_paths), args.concurrency)

    ensure_collection()
    retriever = get_retriever(args.tenant_id)
    manifest = load_manifest(settings.ingestion_manifest_file)

    payload = bm25.load_raw(settings.bm25_index_file)
    existing_corpus: list[str] = payload["corpus"]
    existing_metadatas: list[dict[str, Any]] = payload["metadatas"]

    new_chunks: list[Document] = []
    replaced_sources: set[str] = set()
    pages_total = 0
    processed = 0
    skipped = 0
    failed = 0
    started = time.time()

    for position, (pdf_path, documents, parse_error) in enumerate(_iter_parsed(pdf_paths, args.concurrency), 1):
        key = source_key(pdf_path)
        name = os.path.basename(pdf_path)

        try:
            digest = file_hash(pdf_path)
        except OSError as exc:
            logger.error("Cannot hash %s: %s", name, exc)
            failed += 1
            continue

        if manifest.get(key, {}).get("sha256") == digest:
            skipped += 1
            logger.debug("[%d/%d] Unchanged, skipping: %s", position, len(pdf_paths), name)
            continue

        if parse_error is not None:
            logger.error("[%d/%d] Failed to parse %s: %s", position, len(pdf_paths), name, parse_error)
            failed += 1
            continue

        logger.info("[%d/%d] Indexing %s", position, len(pdf_paths), name)
        try:
            children = index_documents(pdf_path, documents or [], retriever, args.tenant_id)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            logger.exception("[%d/%d] Failed to index %s: %s", position, len(pdf_paths), name, exc)
            failed += 1
            continue

        new_chunks.extend(children)
        replaced_sources.add(name)
        pages_total += len(documents or [])
        processed += 1
        manifest[key] = {
            "sha256": digest,
            "source": name,
            "pages": len(documents or []),
            "chunks": len(children),
            "status": "completed",
            "ingested_at": int(time.time()),
        }
        save_manifest(settings.ingestion_manifest_file, manifest)

    logger.info(
        "Indexed %d PDF(s), skipped %d unchanged, %d failed (%d pages, %.1fs).",
        processed,
        skipped,
        failed,
        pages_total,
        time.time() - started,
    )

    if new_chunks or replaced_sources:
        if settings.database_url:
            write_postgres_fts(settings.database_url, new_chunks, replaced_sources, args.tenant_id)
        else:
            rebuild_bm25(existing_corpus, existing_metadatas, replaced_sources, new_chunks)

    return 1 if failed else 0


def verify(tenant_id: str = "default") -> int:
    """Report drift between the manifest, the vector store, and the keyword index."""
    settings = get_settings()
    manifest = load_manifest(settings.ingestion_manifest_file)
    client = get_qdrant_client()

    expected_chunks = sum(int(entry.get("chunks", 0)) for entry in manifest.values())
    actual_points = count_points(client)
    tenant_points = count_points_for_tenant(tenant_id, client)

    if settings.database_url:
        from src.db.schema import ChildChunkFTS, get_session_factory

        with get_session_factory(settings.database_url)() as session:
            bm25_chunks = session.query(ChildChunkFTS).filter(ChildChunkFTS.tenant_id == tenant_id).count()
    else:
        index = bm25.load_index(settings.bm25_index_file, k=settings.bm25_k)
        bm25_chunks = len(index.corpus) if index is not None else 0

    logger.info("Manifest files  : %d", len(manifest))
    logger.info("Expected chunks : %d", expected_chunks)
    logger.info("Qdrant points   : %d (collection total)", actual_points)
    logger.info("Qdrant points   : %d (tenant=%s)", tenant_points, tenant_id)
    logger.info("Keyword chunks  : %d (tenant=%s)", bm25_chunks, tenant_id)

    drift = False
    if not manifest:
        logger.error("Manifest is empty — nothing has been ingested.")
        drift = True
    if expected_chunks and actual_points != expected_chunks:
        logger.error("Drift: Qdrant holds %d points, the manifest expects %d.", actual_points, expected_chunks)
        drift = True
    if tenant_points != bm25_chunks:
        logger.error(
            "Drift: keyword index holds %d chunks for tenant=%s, Qdrant has %d for that tenant.",
            bm25_chunks,
            tenant_id,
            tenant_points,
        )
        drift = True

    if drift:
        logger.error("Drift detected. Re-run ingestion (add --reset for a clean rebuild).")
        return 1

    logger.info("No drift detected.")
    return 0


def _select_pdfs(args: argparse.Namespace) -> list[str]:
    settings = get_settings()
    if args.pdf:
        return [args.pdf]

    paths = sorted(glob.glob(os.path.join(settings.raw_pdfs_dir, "*.pdf")))[args.offset :]
    if args.limit is not None:
        paths = paths[: args.limit]
    return paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG system.")
    parser.add_argument("--pdf", type=str, default=None, help="Ingest a single PDF instead of the corpus.")
    parser.add_argument("--reset", action="store_true", help="Wipe existing indexes before ingesting.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N PDFs.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N PDFs.")
    parser.add_argument("--tenant-id", type=str, default="default", help="Tenant the documents belong to.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel PDF parsing workers (indexing itself stays serial).",
    )
    parser.add_argument("--verify", action="store_true", help="Report manifest/vector-store drift and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.offset < 0:
        parser.error("--offset cannot be negative")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    settings = get_settings()
    configure_logging(settings.log_level, json_output=False)

    if args.verify:
        return verify(args.tenant_id)
    return run_ingestion(args)


if __name__ == "__main__":
    sys.exit(main())
