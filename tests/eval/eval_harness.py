"""Comprehensive RAG Evaluation Harness
=======================================

Evaluates all dimensions of the RAG pipeline:

  1. Retrieval Quality   — recall@k, MRR, rerank lift (existing)
  2. Chunking Quality    — chunk size distribution, overlap verification
  3. Generation Quality  — end-to-end answer evaluation via the live API
  4. Faithfulness        — hallucination detection on synthetic adversarial queries
  5. Latency Profiling   — per-stage timing breakdown

Usage
-----
    python tests/eval/eval_harness.py                      # run all evaluations
    python tests/eval/eval_harness.py --suite retrieval     # run one suite
    python tests/eval/eval_harness.py --suite chunking
    python tests/eval/eval_harness.py --suite generation
    python tests/eval/eval_harness.py --suite latency
    python tests/eval/eval_harness.py --json                # emit JSON results
    python tests/eval/eval_harness.py --generate-golden 15  # auto-generate 15 golden queries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.config.settings import get_settings  # noqa: E402
from src.observability.logging import configure_logging  # noqa: E402

configure_logging(get_settings().log_level, json_output=False)

GOLDEN_QUERIES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_queries.json")
EVAL_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")


# ═══════════════════════════════════════════════════════════════════════════
# Suite 1: Chunking Quality
# ═══════════════════════════════════════════════════════════════════════════


def eval_chunking() -> dict[str, Any]:
    """Analyse the BM25 index to inspect chunk size distribution and quality."""
    import pickle

    settings = get_settings()
    bm25_path = settings.bm25_index_file

    if not os.path.exists(bm25_path):
        return {"status": "skipped", "reason": f"BM25 index not found at {bm25_path}"}

    with open(bm25_path, "rb") as f:
        data = pickle.load(f)

    chunks = data.get("corpus", [])
    metadata_list = data.get("metadatas", [])

    if not chunks:
        return {"status": "skipped", "reason": "BM25 index contains no chunks"}

    # -- Size distribution --
    sizes = [len(c) for c in chunks]
    size_stats = {
        "count": len(sizes),
        "min": min(sizes),
        "max": max(sizes),
        "mean": round(statistics.mean(sizes), 1),
        "median": round(statistics.median(sizes), 1),
        "stdev": round(statistics.stdev(sizes), 1) if len(sizes) > 1 else 0,
        "p5": round(sorted(sizes)[len(sizes) * 5 // 100], 1),
        "p95": round(sorted(sizes)[len(sizes) * 95 // 100], 1),
    }

    # -- Tiny/huge chunk detection --
    tiny_threshold = 50
    huge_threshold = settings.child_chunk_size * 2
    tiny_chunks = sum(1 for s in sizes if s < tiny_threshold)
    huge_chunks = sum(1 for s in sizes if s > huge_threshold)

    # -- Source distribution --
    sources = Counter(m.get("source", "unknown") for m in metadata_list)
    chunks_per_source = {
        "min": min(sources.values()),
        "max": max(sources.values()),
        "mean": round(statistics.mean(sources.values()), 1),
        "sources": len(sources),
    }

    # -- Overlap detection (sample-based) --
    sample_size = min(200, len(chunks) - 1)
    overlaps_found = 0
    for i in range(sample_size):
        if chunks[i][-40:] in chunks[i + 1][:120]:
            overlaps_found += 1

    overlap_rate = round(overlaps_found / max(sample_size, 1), 4)

    # -- Duplicate detection --
    hashes = [hashlib.sha256(c.encode()).hexdigest() for c in chunks]
    unique_hashes = len(set(hashes))
    duplicate_count = len(hashes) - unique_hashes

    return {
        "status": "ok",
        "chunk_size_distribution": size_stats,
        "target_child_chunk_size": settings.child_chunk_size,
        "target_parent_chunk_size": settings.parent_chunk_size,
        "tiny_chunks_below_50_chars": tiny_chunks,
        "huge_chunks_above_2x_target": huge_chunks,
        "chunks_per_source": chunks_per_source,
        "overlap_rate_sampled": overlap_rate,
        "exact_duplicates": duplicate_count,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Suite 2: Retrieval Quality (extended from run_eval.py)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RetrievalMetrics:
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


def eval_retrieval(k: int = 5) -> dict[str, Any]:
    """Evaluate retrieval quality against golden queries."""
    if not os.path.exists(GOLDEN_QUERIES):
        return {"status": "skipped", "reason": f"No golden query set at {GOLDEN_QUERIES}"}

    with open(GOLDEN_QUERIES, "r", encoding="utf-8") as f:
        payload = json.load(f)
    queries = payload.get("queries", [])

    verified = [q for q in queries if q.get("relevant_sources")]
    if not verified:
        return {"status": "skipped", "reason": "No verified queries in golden set"}

    from src.llm.reranker import get_reranker  # noqa: E402
    from src.retrieval.hybrid_search import get_hybrid_search  # noqa: E402

    retriever = get_hybrid_search()
    reranker = get_reranker()

    baseline = RetrievalMetrics()
    reranked = RetrievalMetrics()
    per_query: list[dict] = []

    for entry in verified:
        query = entry["query"]
        relevant = set(entry["relevant_sources"])
        candidates = retriever.search(query)

        # Baseline (hybrid only)
        rank = _first_rank(candidates[:k], relevant)
        hit = rank is not None
        rr = 1.0 / rank if rank else 0.0
        baseline.evaluated += 1
        baseline.hits += int(hit)
        baseline.reciprocal_ranks.append(rr)

        # Reranked
        top = reranker.rerank(query, candidates, top_n=k) if reranker else candidates[:k]
        rank_r = _first_rank(top, relevant)
        hit_r = rank_r is not None
        rr_r = 1.0 / rank_r if rank_r else 0.0
        reranked.evaluated += 1
        reranked.hits += int(hit_r)
        reranked.reciprocal_ranks.append(rr_r)

        per_query.append(
            {
                "query": query,
                "hybrid_rank": rank,
                "reranked_rank": rank_r,
                "hybrid_hit": hit,
                "reranked_hit": hit_r,
            }
        )

    return {
        "status": "ok",
        "k": k,
        "evaluated": baseline.evaluated,
        "skipped": len(queries) - len(verified),
        "hybrid": {"recall": round(baseline.recall, 4), "mrr": round(baseline.mrr, 4)},
        "reranked": {"recall": round(reranked.recall, 4), "mrr": round(reranked.mrr, 4)},
        "rerank_lift_mrr": round(reranked.mrr - baseline.mrr, 4),
        "per_query": per_query,
    }


def _first_rank(docs: list, relevant: set[str]) -> int | None:
    for i, d in enumerate(docs, 1):
        if d.metadata.get("source") in relevant:
            return i
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Suite 3: End-to-End Generation Quality
# ═══════════════════════════════════════════════════════════════════════════


def eval_generation() -> dict[str, Any]:
    """Run end-to-end generation on golden queries and check quality signals."""
    if not os.path.exists(GOLDEN_QUERIES):
        return {"status": "skipped", "reason": f"No golden query set at {GOLDEN_QUERIES}"}

    with open(GOLDEN_QUERIES, "r", encoding="utf-8") as f:
        payload = json.load(f)
    queries = payload.get("queries", [])
    verified = [q for q in queries if q.get("relevant_sources")]

    if not verified:
        return {"status": "skipped", "reason": "No verified queries"}

    from src.core.rag_pipeline import get_rag_pipeline  # noqa: E402

    pipeline = get_rag_pipeline()

    results = []
    for entry in verified:
        query = entry["query"]
        relevant = set(entry["relevant_sources"])

        t0 = time.perf_counter()

        # Bypass cache to force generation
        original_get = pipeline.cache.get
        pipeline.cache.get = lambda q, **kwargs: None
        try:
            result = pipeline.run(query)
        finally:
            pipeline.cache.get = original_get

        elapsed = time.perf_counter() - t0

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        cited = {s["source"] for s in sources}
        confidence = result.get("confidence_score", 0.0)
        faithful = result.get("faithfulness_passed", False)

        # Check if the correct source was cited
        source_hit = bool(cited & relevant)

        # Check for thinking tag leaks
        thinking_leak = "<thinking>" in answer or "</thinking>" in answer

        results.append(
            {
                "query": query,
                "answer_length": len(answer),
                "num_sources_cited": len(sources),
                "correct_source_cited": source_hit,
                "confidence_score": confidence,
                "faithfulness_passed": faithful,
                "thinking_tag_leak": thinking_leak,
                "latency_seconds": round(elapsed, 3),
                "cached": result.get("cached", False),
            }
        )

        # Pace requests to respect free-tier rate limits (Groq TPM and Cohere RPM)
        if len(verified) > 1:
            time.sleep(6)

    # Aggregate
    n = len(results)
    return {
        "status": "ok",
        "evaluated": n,
        "source_citation_accuracy": round(sum(r["correct_source_cited"] for r in results) / n, 4) if n else 0,
        "faithfulness_pass_rate": round(sum(r["faithfulness_passed"] for r in results) / n, 4) if n else 0,
        "thinking_tag_leaks": sum(r["thinking_tag_leak"] for r in results),
        "avg_confidence_score": round(statistics.mean(r["confidence_score"] for r in results), 4) if n else 0,
        "avg_answer_length": round(statistics.mean(r["answer_length"] for r in results), 1) if n else 0,
        "avg_latency_seconds": round(statistics.mean(r["latency_seconds"] for r in results), 3) if n else 0,
        "per_query": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Suite 4: Latency Profiling
# ═══════════════════════════════════════════════════════════════════════════


def eval_latency() -> dict[str, Any]:
    """Profile per-stage latency of the RAG pipeline."""
    if not os.path.exists(GOLDEN_QUERIES):
        return {"status": "skipped", "reason": "No golden queries"}

    with open(GOLDEN_QUERIES, "r", encoding="utf-8") as f:
        payload = json.load(f)
    queries = payload.get("queries", [])
    verified = [q for q in queries if q.get("relevant_sources")]

    if not verified:
        return {"status": "skipped", "reason": "No verified queries"}

    from src.cache.semantic_cache import get_semantic_cache  # noqa: E402
    from src.core.rag_pipeline import get_rag_pipeline  # noqa: E402
    from src.llm.reranker import get_reranker  # noqa: E402
    from src.retrieval.hybrid_search import get_hybrid_search  # noqa: E402

    retriever = get_hybrid_search()
    reranker = get_reranker()
    cache = get_semantic_cache()
    pipeline = get_rag_pipeline()

    timings: dict[str, list[float]] = defaultdict(list)

    for entry in verified:
        query = entry["query"]

        # Cache lookup
        t0 = time.perf_counter()
        cache.get(query)
        timings["cache_lookup"].append(time.perf_counter() - t0)

        # Hybrid retrieval
        t0 = time.perf_counter()
        candidates = retriever.search(query)
        timings["hybrid_retrieval"].append(time.perf_counter() - t0)

        # Reranking
        if reranker and candidates:
            t0 = time.perf_counter()
            reranker.rerank(query, candidates, top_n=5)
            timings["reranking"].append(time.perf_counter() - t0)

        # Full pipeline (includes generation) - bypass cache to measure true latency
        t0 = time.perf_counter()
        original_get = pipeline.cache.get
        pipeline.cache.get = lambda q, **kwargs: None  # bypass cache lookup
        try:
            pipeline.run(query)
        finally:
            pipeline.cache.get = original_get
        timings["full_pipeline_no_cache"].append(time.perf_counter() - t0)

    result: dict[str, Any] = {"status": "ok", "queries_profiled": len(verified)}
    for stage, times in timings.items():
        result[stage] = {
            "mean_ms": round(statistics.mean(times) * 1000, 1),
            "median_ms": round(statistics.median(times) * 1000, 1),
            "p95_ms": round(sorted(times)[int(len(times) * 0.95)] * 1000, 1)
            if len(times) > 1
            else round(times[0] * 1000, 1),
            "max_ms": round(max(times) * 1000, 1),
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Golden Query Auto-Generator
# ═══════════════════════════════════════════════════════════════════════════


def generate_golden_queries(count: int = 15) -> dict[str, Any]:
    """Auto-generate golden queries by sampling real chunks from the BM25 index."""
    import pickle
    import random

    settings = get_settings()
    bm25_path = settings.bm25_index_file

    if not os.path.exists(bm25_path):
        return {"status": "error", "reason": f"BM25 index not found at {bm25_path}"}

    with open(bm25_path, "rb") as f:
        data = pickle.load(f)

    chunks = data.get("corpus", [])
    metadata_list = data.get("metadatas", [])

    if not chunks:
        return {"status": "error", "reason": "No chunks in BM25 index"}

    # Group chunks by source
    by_source: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for chunk, meta in zip(chunks, metadata_list):
        source = meta.get("source", "unknown")
        if len(chunk) > 200:  # skip very short chunks
            by_source[source].append((chunk, meta))

    if not by_source:
        return {"status": "error", "reason": "No usable chunks found"}

    # Sample: pick chunks from different sources
    sources = list(by_source.keys())
    random.shuffle(sources)

    sampled: list[dict[str, Any]] = []
    for source in sources:
        if len(sampled) >= count:
            break
        source_chunks = by_source[source]
        # Pick 1-2 chunks per source
        picks = random.sample(source_chunks, min(2, len(source_chunks)))
        for chunk_text, meta in picks:
            if len(sampled) >= count:
                break
            # Extract a meaningful snippet for the user to write a query about
            snippet = chunk_text[:500].strip()
            sampled.append(
                {
                    "source": source,
                    "page": meta.get("page"),
                    "snippet": snippet,
                    "suggested_query": "",  # user fills this in
                }
            )

    # Write template
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_queries_template.json")
    template = {
        "_README": [
            "Auto-generated golden query template.",
            "For each entry, read the snippet and write a query that the snippet answers.",
            "Then copy the finished entries into golden_queries.json.",
        ],
        "queries": [
            {
                "query": s["suggested_query"] or f"[WRITE A QUESTION ABOUT: {s['source']} page {s['page']}]",
                "relevant_sources": [s["source"]],
                "_snippet_hint": s["snippet"],
            }
            for s in sampled
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    return {
        "status": "ok",
        "generated": len(sampled),
        "output_path": output_path,
        "sources_covered": len(set(s["source"] for s in sampled)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Report Printer
# ═══════════════════════════════════════════════════════════════════════════


def print_report(results: dict[str, Any]) -> None:
    """Pretty-print the evaluation report."""
    print("\n" + "=" * 72)
    print("  QuantRAG Evaluation Report")
    print("=" * 72)

    # Chunking
    if "chunking" in results:
        c = results["chunking"]
        print("\n-- Chunking Quality --")
        if c.get("status") == "ok":
            dist = c["chunk_size_distribution"]
            print(f"  Total chunks        : {dist['count']}")
            print(f"  Size range          : {dist['min']} - {dist['max']} chars")
            print(f"  Mean / Median       : {dist['mean']} / {dist['median']} chars")
            print(f"  Std dev             : {dist['stdev']} chars")
            print(f"  P5 / P95            : {dist['p5']} / {dist['p95']} chars")
            print(f"  Target child size   : {c['target_child_chunk_size']} chars")
            print(f"  Tiny chunks (<50)   : {c['tiny_chunks_below_50_chars']}")
            print(f"  Huge chunks (>2x)   : {c['huge_chunks_above_2x_target']}")
            print(f"  Exact duplicates    : {c['exact_duplicates']}")
            print(f"  Overlap rate        : {c['overlap_rate_sampled']:.1%}")
            src = c["chunks_per_source"]
            print(f"  Sources indexed     : {src['sources']}")
            print(f"  Chunks/source range : {src['min']} - {src['max']} (mean {src['mean']})")
        else:
            print(f"  SKIPPED: {c.get('reason')}")

    # Retrieval
    if "retrieval" in results:
        r = results["retrieval"]
        print("\n-- Retrieval Quality --")
        if r.get("status") == "ok":
            print(f"  Evaluated           : {r['evaluated']} queries (skipped {r['skipped']})")
            print(f"  Hybrid recall@{r['k']}     : {r['hybrid']['recall']:.3f}")
            print(f"  Hybrid MRR          : {r['hybrid']['mrr']:.3f}")
            print(f"  Reranked recall@{r['k']}   : {r['reranked']['recall']:.3f}")
            print(f"  Reranked MRR        : {r['reranked']['mrr']:.3f}")
            print(f"  Rerank lift (MRR)   : {r['rerank_lift_mrr']:+.3f}")
            print()
            for pq in r.get("per_query", []):
                status = "[OK]" if pq["reranked_hit"] else "[FAIL]"
                print(
                    f'    {status} "{pq["query"][:60]}..."  hybrid=#{pq["hybrid_rank"]} → reranked=#{pq["reranked_rank"]}'
                )
        else:
            print(f"  SKIPPED: {r.get('reason')}")

    # Generation
    if "generation" in results:
        g = results["generation"]
        print("\n-- Generation Quality --")
        if g.get("status") == "ok":
            print(f"  Evaluated               : {g['evaluated']} queries")
            print(f"  Source citation accuracy : {g['source_citation_accuracy']:.1%}")
            print(f"  Faithfulness pass rate   : {g['faithfulness_pass_rate']:.1%}")
            print(f"  Thinking tag leaks      : {g['thinking_tag_leaks']}")
            print(f"  Avg confidence score    : {g['avg_confidence_score']:.3f}")
            print(f"  Avg answer length       : {g['avg_answer_length']:.0f} chars")
            print(f"  Avg latency             : {g['avg_latency_seconds']:.2f}s")
        else:
            print(f"  SKIPPED: {g.get('reason')}")

    # Latency
    if "latency" in results:
        latency = results["latency"]
        print("\n-- Latency Profiling --")
        if latency.get("status") == "ok":
            print(f"  Queries profiled    : {latency['queries_profiled']}")
            for stage in ["cache_lookup", "hybrid_retrieval", "reranking", "full_pipeline_no_cache"]:
                if stage in latency:
                    s = latency[stage]
                    print(
                        f"  {stage:20s}: mean={s['mean_ms']:.0f}ms  median={s['median_ms']:.0f}ms  p95={s['p95_ms']:.0f}ms  max={s['max_ms']:.0f}ms"
                    )
        else:
            print(f"  SKIPPED: {latency.get('reason')}")

    print("\n" + "=" * 72)
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

SUITES = {
    "chunking": eval_chunking,
    "retrieval": eval_retrieval,
    "generation": eval_generation,
    "latency": eval_latency,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QuantRAG Comprehensive Evaluation Harness")
    parser.add_argument("--suite", choices=list(SUITES.keys()), help="Run a specific evaluation suite")
    parser.add_argument("--k", type=int, default=5, help="Rank cutoff for retrieval eval")
    parser.add_argument("--json", action="store_true", help="Emit results as JSON")
    parser.add_argument("--generate-golden", type=int, metavar="N", help="Auto-generate N golden query templates")
    parser.add_argument("--save", action="store_true", help="Save results to eval_results.json")
    args = parser.parse_args(argv)

    # Generate golden queries mode
    if args.generate_golden:
        result = generate_golden_queries(args.generate_golden)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["status"] == "ok":
                print(
                    f"\n[OK] Generated {result['generated']} golden query templates across {result['sources_covered']} sources."
                )
                print(f"  Output: {result['output_path']}")
                print("  Edit the file to write real questions, then copy entries to golden_queries.json\n")
            else:
                print(f"\n[FAIL] {result['reason']}\n")
        return 0 if result["status"] == "ok" else 1

    # Run evaluation suites
    results: dict[str, Any] = {}

    if args.suite:
        suite_fn = SUITES[args.suite]
        if args.suite == "retrieval":
            results[args.suite] = suite_fn(k=args.k)
        else:
            results[args.suite] = suite_fn()
    else:
        # Run all suites
        for name, fn in SUITES.items():
            print(f"Running {name} evaluation...")
            try:
                if name == "retrieval":
                    results[name] = fn(k=args.k)
                else:
                    results[name] = fn()
            except Exception as exc:
                results[name] = {"status": "error", "reason": str(exc)}

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results)

    if args.save:
        with open(EVAL_RESULTS, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {EVAL_RESULTS}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
