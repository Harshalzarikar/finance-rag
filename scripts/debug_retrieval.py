#!/usr/bin/env python3
"""Debug retrieval and ranking for a query across tenants (local ops).

Requires a running stack (Postgres, Qdrant, Redis) and ``PYTHONPATH=``.

Example:
    set PYTHONPATH=.
    python scripts/debug_retrieval.py --query "your question" --tenant acme-insurance
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://raguser:ragpassword@localhost:5432/ragdb")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("SEMANTIC_CACHE_ENABLED", "false")
os.environ.setdefault("APP_ENV", "development")

from src.config.settings import get_settings  # noqa: E402
from src.core.rag_pipeline import get_rag_pipeline  # noqa: E402


def debug_tenant(tenant_id: str, query: str, *, top_k: int, run_pipeline: bool) -> None:
    get_rag_pipeline.cache_clear()
    pipe = get_rag_pipeline(tenant_id)
    docs = pipe.retriever.search(query)
    print(f"\n=== tenant={tenant_id} raw_hits={len(docs)} ===")
    if docs:
        first = docs[0]
        print(" first source:", first.metadata.get("source"), "retriever:", first.metadata.get("retriever"))

    resolved = pipe._deduplicate(pipe._resolve_parents(docs))
    reranked = pipe._rerank(query, resolved, top_k)
    for doc in reranked[:5]:
        print(" score", doc.metadata.get("relevance_score"), "src", doc.metadata.get("source"))

    relevant = pipe._relevant(reranked, query)
    print(" after floor:", len(relevant), "floor", get_settings().rerank_min_score)

    if run_pipeline:
        result = pipe.run(query, top_k_rerank=top_k)
        preview = result["answer"][:200].replace("\n", " ")
        print(" answer:", preview)
        print(" prompt_pack:", result.get("prompt_pack_version"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect hybrid retrieval per tenant.")
    parser.add_argument("--query", required=True, help="Question to debug")
    parser.add_argument(
        "--tenant",
        action="append",
        default=["acme-insurance", "default"],
        help="Tenant id (repeatable). Default: acme-insurance and default",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Rerank top-k")
    parser.add_argument("--no-run", action="store_true", help="Skip full pipeline (no LLM call)")
    args = parser.parse_args()

    get_settings.cache_clear()
    for tenant_id in args.tenant:
        debug_tenant(tenant_id, args.query, top_k=args.top_k, run_pipeline=not args.no_run)


if __name__ == "__main__":
    main()
