"""End-to-end RAG orchestration.

Query path, mirroring RAGFlow's stages:

    query -> semantic cache -> hybrid recall (BM25 + dense)
          -> parent resolution -> cross-document dedup
          -> cross-encoder rerank -> grounded generation

The pipeline is assembled from cached factories so that a dependency being down
is a reported, degraded state rather than a process that refuses to start.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.stores import BaseStore

from src.cache.semantic_cache import SemanticCache, get_semantic_cache
from src.config.settings import get_settings
from src.core.faithfulness_guard import FaithfulnessGuard
from src.core.query_guard import QueryGuard, QueryGuardResult
from src.llm.generator import get_llm
from src.llm.reranker import CohereReranker, get_reranker
from src.observability.metrics import rag_counters
from src.retrieval.hybrid_search import HybridSearchService, get_hybrid_search
from src.retrieval.vector_store import get_docstore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Quantitative Finance & Financial Analyst AI.
Your task is to answer the user's question STRICTLY based on the provided Context.
You must use a Chain-of-Thought process before answering. Enclose your reasoning in <thinking>...</thinking> XML tags.

Within <thinking>:
1. Evaluate if the Context actually contains the information required to answer the Question.
2. Extract exact quotes from the Context that support the answer.
3. If the Context does not contain sufficient information, state this explicitly in the <thinking> block and plan to decline to answer.

After the <thinking> block, provide your final concise answer.
- Never invent or infer facts outside the Context.
- Cite the source documents you relied on, using the [Source: <name>] labels.
- If you lack sufficient information, state clearly that you do not have enough information to answer.

Context:
{context}
"""

# Metadata keys that carry a candidate's retrieval score, preserved when a child
# chunk is replaced by its parent section.
_SCORE_KEYS = ("retriever", "bm25_score", "relevance_score")

# Regex to strip Chain-of-Thought <thinking>...</thinking> blocks from responses.
_THINKING_RE = re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL)

# Returned when retrieval finds nothing relevant enough to answer from. The
# generator is not consulted, because there is nothing to ground an answer in and
# citing the near-misses would imply they support it.
NO_RELEVANT_PASSAGE_ANSWER = (
    "No passage in the indexed corpus is relevant to this question, so I cannot answer it. "
    "The corpus may not cover this topic, or it may not be indexed yet."
)

HALLUCINATION_GUARD_ANSWER = (
    "I'm sorry, but I cannot provide a verified answer based purely on the retrieved context. "
    "The generated response failed the strict hallucination check."
)

QUERY_BLOCKED_ANSWER = (
    "This query has been blocked by the safety filter. "
    "Please rephrase your question to focus on quantitative finance topics."
)


class RAGPipeline:
    """Coordinates retrieval, reranking, generation, and caching."""

    def __init__(
        self,
        retriever: HybridSearchService,
        reranker: CohereReranker,
        generator: Any,
        cache: SemanticCache,
        docstore: BaseStore | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.cache = cache
        self.docstore = docstore
        self.faithfulness_guard = FaithfulnessGuard(llm=generator)
        self.query_guard = QueryGuard()
        self.prompt_template = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, query: str, top_k_rerank: int = 5) -> dict[str, Any]:
        """Answer ``query`` end to end, consulting the semantic cache first."""
        guard_result = self.query_guard.check(query)
        if not guard_result.allowed:
            rag_counters("query_blocked")
            return {
                "answer": QUERY_BLOCKED_ANSWER,
                "sources": [],
                "cached": False,
                "cache_similarity": None,
                "confidence_score": 0.0,
                "faithfulness_passed": False,
            }

        cached = self.cache.get(query)
        if cached is not None:
            return {
                "answer": cached.answer,
                "sources": cached.sources,
                "cached": True,
                "cache_similarity": cached.similarity,
                "confidence_score": 1.0,
                "faithfulness_passed": True,
            }

        documents = self._retrieve(query)
        reranked = self._relevant(self._rerank(query, documents, top_k_rerank))
        if not reranked:
            rag_counters("declined")
            return {
                "answer": NO_RELEVANT_PASSAGE_ANSWER,
                "sources": [],
                "cached": False,
                "cache_similarity": None,
                "confidence_score": 0.0,
                "faithfulness_passed": False,
            }

        raw_answer = self._generate(query, reranked)
        answer = self._strip_thinking(raw_answer)
        context = self._build_context(reranked)
        confidence = self._compute_confidence(reranked)

        if get_settings().enable_faithfulness_guard:
            if not self.faithfulness_guard.check(query, context, answer):
                rag_counters("faithfulness_blocked")
                return {
                    "answer": HALLUCINATION_GUARD_ANSWER,
                    "sources": [],
                    "cached": False,
                    "cache_similarity": None,
                    "confidence_score": 0.0,
                    "faithfulness_passed": False,
                }

        sources = self._build_sources(reranked)
        rag_counters("answered")

        self.cache.set(query, answer, sources)
        return {
            "answer": answer,
            "sources": sources,
            "cached": False,
            "cache_similarity": None,
            "confidence_score": confidence,
            "faithfulness_passed": True,
        }

    def stream(self, query: str, top_k_rerank: int = 5) -> Iterator[dict[str, Any]]:
        """Yield the answer incrementally as ``sources``/``token``/``done`` events.

        Sources are emitted before the first token so the UI can render citations
        immediately. A cache hit is replayed as a single token.
        """
        guard_result = self.query_guard.check(query)
        if not guard_result.allowed:
            rag_counters("query_blocked")
            yield {"type": "sources", "sources": [], "cached": False}
            yield {"type": "token", "value": QUERY_BLOCKED_ANSWER}
            yield {"type": "done", "cached": False, "confidence_score": 0.0, "faithfulness_passed": False}
            return

        cached = self.cache.get(query)
        if cached is not None:
            yield {"type": "sources", "sources": cached.sources, "cached": True}
            yield {"type": "token", "value": cached.answer}
            yield {"type": "done", "cached": True, "confidence_score": 1.0, "faithfulness_passed": True}
            return

        documents = self._retrieve(query)
        reranked = self._relevant(self._rerank(query, documents, top_k_rerank))
        sources = self._build_sources(reranked)
        confidence = self._compute_confidence(reranked)
        yield {"type": "sources", "sources": sources, "cached": False}

        if not reranked:
            rag_counters("declined")
            yield {"type": "token", "value": NO_RELEVANT_PASSAGE_ANSWER}
            yield {"type": "done", "cached": False, "confidence_score": 0.0, "faithfulness_passed": False}
            return

        context = self._build_context(reranked)
        chunks: list[str] = []
        for part in self._chain().stream({"context": context, "question": query}):
            token = self._coerce_content(part.content)
            if token:
                chunks.append(token)
                yield {"type": "token", "value": token}

        raw_answer = "".join(chunks)
        answer = self._strip_thinking(raw_answer)

        if get_settings().enable_faithfulness_guard and answer:
            if not self.faithfulness_guard.check(query, context, answer):
                rag_counters("faithfulness_blocked")
                yield {"type": "error", "detail": "\n\n**Warning: The generated answer failed the strict faithfulness check and may contain hallucinations.**"}
                yield {"type": "done", "cached": False, "confidence_score": 0.0, "faithfulness_passed": False}
                return

        if answer:
            rag_counters("answered")
            self.cache.set(query, answer, sources)
        yield {"type": "done", "cached": False, "confidence_score": confidence, "faithfulness_passed": True}

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _retrieve(self, query: str) -> list[Document]:
        settings = get_settings()
        candidates = self.retriever.search(query)
        resolved = self._resolve_parents(candidates)
        deduplicated = self._deduplicate(resolved)
        logger.info(
            "Retrieved %d candidates → %d after parent resolution and dedup.",
            len(candidates),
            len(deduplicated),
        )
        return deduplicated[: settings.rerank_max_candidates]

    def _resolve_parents(self, documents: list[Document]) -> list[Document]:
        """Replace BM25 child chunks with their parent sections.

        Dense retrieval already returns parent sections, so without this the fusion
        step would mix small keyword chunks with large vector sections. Every child
        chunk carries the ``doc_id`` of its parent, which is what we look up.
        """
        if self.docstore is None:
            return documents

        parent_ids = [doc.metadata["doc_id"] for doc in documents if doc.metadata.get("doc_id")]
        if not parent_ids:
            return documents

        parents = self.docstore.mget(parent_ids)
        by_id = {parent_id: parent for parent_id, parent in zip(parent_ids, parents, strict=True) if parent}

        resolved: list[Document] = []
        for document in documents:
            parent = by_id.get(document.metadata.get("doc_id", ""))
            if parent is None:
                resolved.append(document)
                continue
            # Keep the parent's text and identity, but carry the scores across so
            # the source list can still report why a passage was selected.
            metadata = {**parent.metadata, **document.metadata}
            resolved.append(Document(page_content=parent.page_content, metadata=metadata))
        return resolved

    @staticmethod
    def _deduplicate(documents: list[Document]) -> list[Document]:
        """Collapse repeated chunks so one paper cannot fill the whole context window."""
        seen: set[str] = set()
        unique: list[Document] = []
        for document in documents:
            fingerprint = hashlib.sha256(
                f"{document.metadata.get('source')}|{document.metadata.get('page')}|{document.page_content}".encode()
            ).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append(document)
        return unique

    def _rerank(self, query: str, documents: list[Document], top_k_rerank: int) -> list[Document]:
        if not documents:
            return []
        return self.reranker.rerank(query, documents, top_n=top_k_rerank)

    @staticmethod
    def _relevant(documents: list[Document]) -> list[Document]:
        """Keep only passages the reranker judged relevant enough to cite.

        Without this, a query the corpus cannot answer still produced a full list of
        citations — the least-irrelevant passages — which reads as evidence for an
        answer that does not exist.

        If no document carries a reranker score (reranking disabled or failed), the
        floor cannot be applied and everything is kept, preserving the previous
        behaviour when the reranker is unavailable.
        """
        scored = [doc for doc in documents if doc.metadata.get("relevance_score") is not None]
        if not scored:
            return documents

        floor = get_settings().rerank_min_score
        relevant = [doc for doc in scored if float(doc.metadata["relevance_score"]) >= floor]

        if len(relevant) != len(scored):
            logger.info(
                "Dropped %d/%d passages below the relevance floor (%.2f).",
                len(scored) - len(relevant),
                len(scored),
                floor,
            )
        return relevant

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _chain(self) -> Any:
        """Prompt-plus-model chain. A seam so tests can supply a fake model."""
        return self.prompt_template | self.generator

    def _generate(self, query: str, documents: list[Document]) -> str:
        context = self._build_context(documents)
        response = self._chain().invoke({"context": context, "question": query})
        return self._coerce_content(response.content)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove Chain-of-Thought ``<thinking>...</thinking>`` blocks from the output."""
        return _THINKING_RE.sub("", text).strip()

    @staticmethod
    def _compute_confidence(documents: list[Document]) -> float:
        """Derive a 0–1 confidence score from the reranker relevance scores."""
        scores = [
            float(doc.metadata["relevance_score"])
            for doc in documents
            if doc.metadata.get("relevance_score") is not None
        ]
        if not scores:
            return 0.5  # unknown confidence when reranker is unavailable
        return round(min(1.0, sum(scores) / len(scores)), 4)

    @staticmethod
    def _build_context(documents: list[Document]) -> str:
        if not documents:
            return "(no relevant context was retrieved)"
        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'Unknown')}"
            + (f", page {doc.metadata['page']}" if doc.metadata.get("page") else "")
            + f"]\n{doc.page_content}"
            for doc in documents
        )

    @staticmethod
    def _build_sources(documents: list[Document]) -> list[dict[str, Any]]:
        return [
            {
                "source": doc.metadata.get("source") or "Unknown",
                "page": doc.metadata.get("page"),
                "score": doc.metadata.get("relevance_score"),
                "snippet": doc.page_content[:300].strip(),
            }
            for doc in documents
        ]

    @staticmethod
    def _coerce_content(content: Any) -> str:
        """Normalise provider responses, some of which return content blocks."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
        return str(content)


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    """Assemble the pipeline once per process."""
    logger.info("Assembling RAG pipeline.")
    return RAGPipeline(
        retriever=get_hybrid_search(),
        reranker=get_reranker(),
        generator=get_llm(),
        cache=get_semantic_cache(),
        docstore=get_docstore(),
    )
