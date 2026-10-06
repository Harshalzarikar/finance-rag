"""Tests for dynamic prompt assembly."""

from __future__ import annotations

from langchain_core.documents import Document

from src.core.prompt_builder import (
    CorpusProfile,
    QueryIntent,
    build_system_prompt,
    classify_query_intent,
    infer_corpus_profile,
    wants_comprehensive_answer,
)


def test_classify_comprehensive_intent():
    assert classify_query_intent("tell me all details about Jane Doe") == QueryIntent.COMPREHENSIVE


def test_classify_compare_intent():
    assert classify_query_intent("compare delta hedging vs gamma scalping") == QueryIntent.COMPARE


def test_wants_comprehensive_for_list_queries():
    assert wants_comprehensive_answer("list all skills on the resume")


def test_infer_resume_corpus():
    docs = [
        Document(
            page_content="Experience\nAI Engineer at Acme",
            metadata={"source": "candidate_resume.pdf", "page": 1},
        )
    ]
    assert infer_corpus_profile(docs) == CorpusProfile.RESUME


def test_build_system_prompt_includes_context_and_style():
    docs = [Document(page_content="Fact A", metadata={"source": "paper.pdf", "page": 2})]
    prompt = build_system_prompt(
        query="summarize the paper",
        context="[Source: paper.pdf, page 2]\nFact A",
        documents=docs,
    )
    assert "summarize" in prompt.lower() or QueryIntent.SUMMARIZE.value in prompt
    assert "[Source: paper.pdf" in prompt
    assert "Fact A" in prompt
    assert "<thinking>" in prompt


def test_tenant_instructions_appear_in_prompt():
    docs = [Document(page_content="Fact A", metadata={"source": "doc.pdf"})]
    prompt = build_system_prompt(
        query="what is fact A?",
        context="[Source: doc.pdf]\nFact A",
        documents=docs,
        tenant_instructions="Always answer in bullet points.",
    )
    assert "Organization instructions" in prompt
    assert "bullet points" in prompt


def test_document_type_metadata_overrides_corpus():
    docs = [
        Document(
            page_content="Some text",
            metadata={"source": "file.pdf", "document_type": "resume"},
        )
    ]
    assert infer_corpus_profile(docs) == CorpusProfile.RESUME
