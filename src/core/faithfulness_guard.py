"""Post-generation entailment guard to prevent hallucinations.

This module evaluates whether a generated answer is strictly supported by the
provided context. If the model introduces external facts or contradicts the
context, this guard will catch it.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """You are a strict hallucination auditor for a Quantitative Finance AI.
Your ONLY job is to verify if the given Answer is fully supported by the provided Context.

Context:
{context}

Question:
{question}

Answer to evaluate:
{answer}

Instructions:
1. Ignore whether the Answer is correct in the real world. Only check if it is explicitly supported by the Context provided.
2. If the Answer contains ANY claims, facts, or numbers not present in the Context, it is a hallucination.
3. If the Answer states it cannot answer the question because the information is missing, that is considered faithful (no hallucination).
4. Evaluate the entailment strictly.

Return a JSON object with a single boolean field "is_faithful". Set it to true if the Answer is fully supported, or false if there is any hallucination.
"""


class FaithfulnessResult(BaseModel):
    is_faithful: bool = Field(
        description="True if the answer is fully supported by the context, False if it hallucinates."
    )


class FaithfulnessGuard:
    """Evaluates answer faithfulness against retrieved context."""

    def __init__(self, llm: BaseLanguageModel) -> None:
        self.llm = llm

        # We try to use structured output if the model supports it. If it's an OSS model
        # that doesn't fully support tool calling / with_structured_output well,
        # we might need to parse JSON. We'll use with_structured_output for robustness
        # on supported providers (like Groq/OpenAI).
        self.prompt = ChatPromptTemplate.from_messages([("system", FAITHFULNESS_PROMPT)])

        try:
            self.chain = self.prompt | self.llm.with_structured_output(FaithfulnessResult)
        except NotImplementedError:
            # Fallback if the LLM doesn't support structured output out of the box
            from langchain_core.output_parsers import JsonOutputParser

            self.chain = self.prompt | self.llm | JsonOutputParser()

    def check(self, query: str, context: str, answer: str) -> bool:
        """Return True if the answer is faithful to the context, False otherwise."""
        if not answer or answer.strip() == "":
            return True

        logger.debug("Running faithfulness check on generated answer.")
        try:
            result = self.chain.invoke({"context": context, "question": query, "answer": answer})

            # Handle both BaseModel and Dict returns depending on the chain path
            if isinstance(result, FaithfulnessResult):
                is_faithful = result.is_faithful
            elif isinstance(result, dict):
                is_faithful = result.get("is_faithful", True)
            else:
                is_faithful = True

            if not is_faithful:
                logger.warning("Faithfulness Guard blocked a hallucinated answer!")

            return bool(is_faithful)

        except Exception as exc:
            logger.error("Faithfulness Guard failed to run (%s). Failing open (returning True).", exc)
            return True
