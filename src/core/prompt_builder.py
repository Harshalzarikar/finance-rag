"""Dynamic system prompts for grounded generation.

Builds a fresh system prompt per request from:

- **Prompt pack** (versioned JSON under ``src/core/prompts/``) — persona, style, grounding
- **Query intent** — concise, comprehensive, list, compare, summarize
- **Corpus profile** — inferred from retrieved document metadata/content
- **Tenant + deployment overlays** — ``custom_instructions`` per org, ``RAG_PERSONA`` env globally
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document

from src.config.settings import get_settings
from src.core.prompt_loader import PromptPack, get_prompt_pack, resolve_pack_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent classification (rule-based; fast and deterministic for tests/ops)
# ---------------------------------------------------------------------------

_COMPREHENSIVE_MARKERS = (
    "all details",
    "full details",
    "everything about",
    "complete profile",
    "tell me more",
    "more about",
    "in detail",
    "comprehensive",
    "full background",
    "work history",
    "full resume",
    "entire resume",
)

_LIST_MARKERS = (
    "list all",
    "list every",
    "enumerate",
    "bullet",
    "what are the",
    "name all",
)

_COMPARE_MARKERS = (
    "compare",
    "difference between",
    "differences between",
    " vs ",
    " versus ",
    "contrast ",
)

_SUMMARIZE_MARKERS = (
    "summarize",
    "summary of",
    "overview of",
    "tl;dr",
    "in brief",
)


class QueryIntent(str, Enum):
    CONCISE = "concise"
    COMPREHENSIVE = "comprehensive"
    LIST = "list"
    COMPARE = "compare"
    SUMMARIZE = "summarize"


class CorpusProfile(str, Enum):
    RESEARCH = "research"
    RESUME = "resume"
    POLICY = "policy"
    GENERAL = "general"


@dataclass(frozen=True)
class PromptPlan:
    intent: QueryIntent
    corpus: CorpusProfile
    pack_id: str
    pack_version: str


def classify_query_intent(query: str) -> QueryIntent:
    """Map a user question to a response-shape intent."""
    q = query.lower()
    if any(m in q for m in _COMPREHENSIVE_MARKERS):
        return QueryIntent.COMPREHENSIVE
    if any(m in q for m in _COMPARE_MARKERS):
        return QueryIntent.COMPARE
    if any(m in q for m in _LIST_MARKERS):
        return QueryIntent.LIST
    if any(m in q for m in _SUMMARIZE_MARKERS):
        return QueryIntent.SUMMARIZE
    if any(m in q for m in ("experience", "education", "skills", " resume", "resume ", " cv", "cv ")):
        return QueryIntent.COMPREHENSIVE
    return QueryIntent.CONCISE


def wants_comprehensive_answer(query: str) -> bool:
    """Whether semantic cache should be bypassed for this query."""
    intent = classify_query_intent(query)
    return intent in {QueryIntent.COMPREHENSIVE, QueryIntent.LIST, QueryIntent.COMPARE}


def infer_corpus_profile(documents: list[Document]) -> CorpusProfile:
    """Guess document type from filenames and snippet cues."""
    if not documents:
        return CorpusProfile.GENERAL

    blob = " ".join(f"{doc.metadata.get('source', '')} {doc.page_content[:800]}" for doc in documents[:5]).lower()

    doc_type = _metadata_corpus_hint(documents)
    if doc_type is not None:
        return doc_type

    if _RESUME_RE.search(blob):
        return CorpusProfile.RESUME
    if _POLICY_RE.search(blob):
        return CorpusProfile.POLICY
    if _RESEARCH_RE.search(blob):
        return CorpusProfile.RESEARCH
    return CorpusProfile.GENERAL


def _metadata_corpus_hint(documents: list[Document]) -> CorpusProfile | None:
    """Optional upload tag: metadata document_type on ingested chunks."""
    mapping = {
        "research": CorpusProfile.RESEARCH,
        "resume": CorpusProfile.RESUME,
        "policy": CorpusProfile.POLICY,
        "general": CorpusProfile.GENERAL,
    }
    for doc in documents[:5]:
        raw = doc.metadata.get("document_type") or doc.metadata.get("doc_type")
        if not raw:
            continue
        key = str(raw).lower().strip()
        if key in mapping:
            return mapping[key]
    return None


_RESUME_RE = re.compile(
    r"(resume|curriculum vitae|\bcv\b|linkedin\.com|github\.com|"
    r"work experience|professional experience|technical skills)",
    re.I,
)
_POLICY_RE = re.compile(
    r"(policy|coverage|insured|premium|claim|underwriting|endorsement|policyholder)",
    re.I,
)
_RESEARCH_RE = re.compile(
    r"(abstract|volatility|black[- ]scholes|garch|stochastic|arxiv|"
    r"quantitative finance|derivatives|portfolio)",
    re.I,
)


def plan_prompt(
    query: str,
    documents: list[Document],
    *,
    tenant_pack_id: str | None = None,
) -> PromptPlan:
    settings = get_settings()
    pack_id = resolve_pack_id(tenant_pack_id=tenant_pack_id, default_pack_id=settings.prompt_pack_id)
    pack = get_prompt_pack(pack_id)
    return PromptPlan(
        intent=classify_query_intent(query),
        corpus=infer_corpus_profile(documents),
        pack_id=pack.pack_id,
        pack_version=pack.version,
    )


def cache_scope_for_prompt(pack_version: str, tenant_id: str, tenant_instructions: str) -> str:
    """Scope semantic cache so prompt or tenant instruction changes do not reuse stale answers."""
    settings = get_settings()
    persona = settings.rag_persona.strip()
    digest = hashlib.sha256(f"{pack_version}|{tenant_id}|{persona}|{tenant_instructions}".encode()).hexdigest()[:16]
    return f"prompt:{digest}"


def build_system_prompt(
    *,
    query: str,
    context: str,
    documents: list[Document],
    tenant_instructions: str = "",
    tenant_pack_id: str | None = None,
) -> str:
    """Compose the full system message for one generation call."""
    plan = plan_prompt(query, documents, tenant_pack_id=tenant_pack_id)
    pack = get_prompt_pack(plan.pack_id)
    settings = get_settings()

    persona_line = pack.persona.get(plan.corpus.value, pack.persona.get("general", ""))
    style_line = pack.response_style.get(plan.intent.value, pack.response_style.get("concise", ""))

    parts = [
        persona_line,
        pack.grounding_rules,
        f"Response style for this question ({plan.intent.value}): {style_line}",
    ]

    deployment_persona = settings.rag_persona.strip()
    if deployment_persona:
        parts.insert(1, f"Platform instructions:\n{deployment_persona}")

    org_text = tenant_instructions.strip()
    if org_text:
        parts.insert(1, f"Organization instructions:\n{org_text}")

    parts.append(f"Context:\n{context}")
    assembled = "\n\n".join(parts)
    logger.debug(
        "Built system prompt pack=%s@%s intent=%s corpus=%s",
        plan.pack_id,
        plan.pack_version,
        plan.intent.value,
        plan.corpus.value,
    )
    return assembled


def build_generation_plan(
    query: str,
    documents: list[Document],
    *,
    tenant_pack_id: str | None = None,
) -> PromptPlan:
    """Public helper for retrieval tuning (cache, top-k) outside generation."""
    return plan_prompt(query, documents, tenant_pack_id=tenant_pack_id)


def effective_prompt_pack(tenant_pack_id: str | None = None) -> PromptPack:
    """Resolve the pack that would be used for a tenant."""
    settings = get_settings()
    pack_id = resolve_pack_id(tenant_pack_id=tenant_pack_id, default_pack_id=settings.prompt_pack_id)
    return get_prompt_pack(pack_id)
