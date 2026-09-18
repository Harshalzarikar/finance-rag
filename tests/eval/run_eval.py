"""Retrieval evaluation harness.

Measures retrieval quality against a set of golden queries, independently of
generation quality:

    recall@k   fraction of queries whose relevant document appears in the top k
    MRR        mean reciprocal rank of the first relevant document
    rerank lift   MRR after cross-encoder reranking minus MRR before it

This intentionally runs offline against the live index (no LLM calls), so it is
deterministic and cheap enough to run on every ingestion.

Usage
-----
    python tests/eval/run_eval.py                     # uses tests/eval/golden_queries.json
    python tests/eval/run_eval.py --k 5 --queries path/to/queries.json
    python tests/eval/run_eval.py --no-rerank         # skip the rerank comparison
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.config.settings import get_settings  # noqa: E402
from src.llm.reranker import get_reranker  # noqa: E402
from src.observability.logging import configure_logging  # noqa: E402
from src.retrieval.hybrid_search import get_hybrid_search  # noqa: E402

DEFAULT_QUERIES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_queries.json")
EXAMPLE_QUERIES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_queries.example.json")


@dataclass
class Metrics:
    evaluated: int = 0
    skipped: int = 0
    hits: int = 0
    reciprocal_ranks: list[float] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.hits / self.evaluated if self.evaluated else 0.0

    @property
    def mrr(self) -> float:
        return sum(self.reciprocal_ranks) / self.evaluated if self.evaluated else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluated": self.evaluated,
            "skipped": self.skipped,
            "recall": round(self.recall, 4),
            "mrr": round(self.mrr, 4),
        }


def load_queries(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    queries = payload.get("queries") if isinstance(payload, dict) else payload
    if not isinstance(queries, list):
        raise ValueError(f"{path} must contain a list of queries (or an object with a 'queries' list).")
    return queries


def first_relevant_rank(documents: list[Any], relevant: set[str]) -> int | None:
    """1-based rank of the first document whose source is relevant, or None."""
    for rank, document in enumerate(documents, start=1):
        if document.metadata.get("source") in relevant:
            return rank
    return None


def score(documents: list[Any], relevant: set[str], k: int) -> tuple[bool, float]:
    rank = first_relevant_rank(documents[:k], relevant)
    if rank is None:
        return False, 0.0
    return True, 1.0 / rank


def evaluate(queries: list[dict[str, Any]], k: int, with_rerank: bool) -> tuple[Metrics, Metrics]:
    retriever = get_hybrid_search()
    reranker = get_reranker() if with_rerank else None

    baseline = Metrics()
    reranked = Metrics()

    for entry in queries:
        query = str(entry.get("query", "")).strip()
        relevant = set(entry.get("relevant_sources") or [])
        if not query or not relevant:
            baseline.skipped += 1
            reranked.skipped += 1
            continue

        candidates = retriever.search(query)

        hit, reciprocal_rank = score(candidates, relevant, k)
        baseline.evaluated += 1
        baseline.hits += int(hit)
        baseline.reciprocal_ranks.append(reciprocal_rank)

        if reranker is None:
            continue

        top = reranker.rerank(query, candidates, top_n=k)
        hit, reciprocal_rank = score(top, relevant, k)
        reranked.evaluated += 1
        reranked.hits += int(hit)
        reranked.reciprocal_ranks.append(reciprocal_rank)

    return baseline, reranked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against golden queries.")
    parser.add_argument("--queries", default=DEFAULT_QUERIES, help="Path to the golden query set.")
    parser.add_argument("--k", type=int, default=5, help="Rank cutoff for recall@k.")
    parser.add_argument("--no-rerank", action="store_true", help="Skip the rerank comparison.")
    parser.add_argument("--json", action="store_true", help="Emit results as JSON.")
    args = parser.parse_args(argv)

    configure_logging(get_settings().log_level, json_output=False)

    queries_path = args.queries
    if not os.path.exists(queries_path):
        if os.path.exists(EXAMPLE_QUERIES) and args.queries == DEFAULT_QUERIES:
            print(f"No golden query set at {queries_path}.")
            print(f"Copy {EXAMPLE_QUERIES} there and fill in verified relevant_sources.")
            print("See the _README field in that file for the workflow.")
            return 1
        print(f"Query set not found: {queries_path}")
        return 1

    queries = load_queries(queries_path)
    baseline, reranked = evaluate(queries, args.k, with_rerank=not args.no_rerank)

    results: dict[str, Any] = {"queries_file": queries_path, "k": args.k, "hybrid": baseline.as_dict()}

    if not args.no_rerank:
        results["reranked"] = reranked.as_dict()
        results["rerank_lift_mrr"] = round(reranked.mrr - baseline.mrr, 4)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"\nGolden query set : {queries_path}")
    print(f"Rank cutoff      : k={args.k}")
    print(f"Evaluated        : {baseline.evaluated} (skipped {baseline.skipped} without verified sources)")
    print()
    print(f"Hybrid recall@{args.k}    : {baseline.recall:.3f}")
    print(f"Hybrid MRR       : {baseline.mrr:.3f}")
    if not args.no_rerank:
        print(f"Reranked recall@{args.k}  : {reranked.recall:.3f}")
        print(f"Reranked MRR     : {reranked.mrr:.3f}")
        print(f"Rerank lift (MRR): {reranked.mrr - baseline.mrr:+.3f}")
    print()

    if baseline.evaluated == 0:
        print("No queries were evaluated — fill in relevant_sources and re-run.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
