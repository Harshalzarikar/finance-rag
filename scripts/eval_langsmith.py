"""LangSmith evaluation harness.

Runs the RAG pipeline over the golden query set, uploads each run to LangSmith, and
scores it with custom evaluators (source-citation accuracy and the pipeline's own
faithfulness flag).

The golden set is the same ``tests/eval/golden_queries.json`` used by the offline
``run_eval.py`` harness, so retrieval and generation evals share one source of truth.

Requirements
------------
    LANGSMITH_API_KEY (and optionally LANGSMITH_TRACING=true) must be set.

Usage
-----
    python scripts/eval_langsmith.py                            # upload golden set + evaluate
    python scripts/eval_langsmith.py --queries path/to/queries.json
    python scripts/eval_langsmith.py --dataset quant-finance-rag  # custom dataset name
    python scripts/eval_langsmith.py --no-upload                 # evaluate the existing dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from langsmith import Client, evaluate  # noqa: E402
from langsmith.evaluation import EvaluationResult, EvaluationResults, RunEvaluator  # noqa: E402
from langsmith.schemas import Example, Run  # noqa: E402

from src.config.settings import get_settings  # noqa: E402
from src.core.rag_pipeline import get_rag_pipeline  # noqa: E402

DEFAULT_QUERIES = ROOT / "tests" / "eval" / "golden_queries.json"
DEFAULT_DATASET = "quant-finance-rag-golden"


def load_queries(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    queries = payload.get("queries") if isinstance(payload, dict) else payload
    return [q for q in queries if q.get("query") and q.get("relevant_sources")]


def make_target():
    """Return a target that runs the full pipeline (cache bypassed) for one question."""
    pipeline = get_rag_pipeline()

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = str(inputs["question"])
        original_get = pipeline.cache.get
        pipeline.cache.get = lambda *args, **kwargs: None  # force the full retrieval+generation path
        try:
            result = pipeline.run(question)
        finally:
            pipeline.cache.get = original_get

        sources = result.get("sources", [])
        return {
            "answer": result.get("answer", ""),
            "cited_sources": [s.get("source") for s in sources],
            "contexts": [s.get("snippet", "") for s in sources],
            "faithfulness_passed": result.get("faithfulness_passed", False),
            "cached": result.get("cached", False),
        }

    return target


class SourceCitationEvaluator(RunEvaluator):
    """Scores whether the pipeline cited at least one source known to be relevant."""

    @property
    def feedback_keys(self) -> list[str]:
        return ["source_citation_accuracy"]

    def evaluate_run(self, run: Run, example: Example | None = None) -> EvaluationResult | EvaluationResults:
        cited = set(run.outputs.get("cited_sources", [])) if run.outputs else set()
        relevant = set((example.outputs or {}).get("relevant_sources", [])) if example else set()
        hit = bool(cited & relevant)
        return EvaluationResult(
            key="source_citation_accuracy",
            score=1.0 if hit else 0.0,
            comment=f"cited {len(cited)} source(s), {len(cited & relevant)} of them relevant",
        )


class FaithfulnessEvaluator(RunEvaluator):
    """Surfaces the pipeline's own post-generation faithfulness audit as a score."""

    @property
    def feedback_keys(self) -> list[str]:
        return ["faithfulness_passed"]

    def evaluate_run(self, run: Run, example: Example | None = None) -> EvaluationResult | EvaluationResults:
        del example
        passed = bool(run.outputs.get("faithfulness_passed")) if run.outputs else False
        return EvaluationResult(key="faithfulness_passed", score=1.0 if passed else 0.0)


def ensure_dataset(client: Client, name: str, examples: list[dict[str, Any]]) -> None:
    """Create the dataset (if missing) and replace its examples with the golden set."""
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
        for existing in client.list_examples(dataset_id=dataset.id):
            client.delete_example(existing.id)
    else:
        dataset = client.create_dataset(dataset_name=name, description="Golden queries for quant-finance-rag.")

    client.create_examples(examples=examples, dataset_id=dataset.id)
    print(f"Uploaded {len(examples)} examples to dataset '{name}'.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with LangSmith.")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES), help="Path to the golden query set JSON.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="LangSmith dataset name.")
    parser.add_argument("--no-upload", action="store_true", help="Evaluate the existing dataset without re-uploading.")
    parser.add_argument("--experiment-prefix", default="quant-finance-rag", help="Prefix for the experiment name.")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.langsmith_api_key:
        print("LANGSMITH_API_KEY is not set — evaluation requires LangSmith.", file=sys.stderr)
        return 1

    from src.observability.tracing import configure_langsmith

    configure_langsmith(settings)
    client = Client(api_key=settings.langsmith_api_key)

    if not args.no_upload:
        if not os.path.exists(args.queries):
            print(f"Golden query set not found: {args.queries}", file=sys.stderr)
            return 1
        queries = load_queries(args.queries)
        if not queries:
            print("No verified queries (query + relevant_sources) in the golden set.", file=sys.stderr)
            return 1
        examples = [
            {"inputs": {"question": q["query"]}, "outputs": {"relevant_sources": q["relevant_sources"]}}
            for q in queries
        ]
        ensure_dataset(client, args.dataset, examples)

    results = evaluate(
        make_target(),
        data=args.dataset,
        evaluators=[SourceCitationEvaluator(), FaithfulnessEvaluator()],
        experiment_prefix=args.experiment_prefix,
        max_concurrency=1,
    )

    print(f"\nEvaluation complete: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
