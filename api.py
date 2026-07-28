import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

# Import RAG components from our backend
from core import cohere_rerank, dynamic_retrieve, setup_rag

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quantitative Finance RAG API",
    description="Production REST API for querying the Agentic RAG architecture.",
)

prompt_template = PromptTemplate.from_template(
    "You are an AI assistant answering questions based on the provided context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer the question using ONLY the provided context. If the answer is not in the context, say 'I don't know'."
)


# Pydantic models for request and response validation
class ChatRequest(BaseModel):
    query: str


class SourceDoc(BaseModel):
    source: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]


# Global cache for components to avoid re-initializing on every request
_GLOBAL_RAG_COMPONENTS: Optional[Tuple[Dict[str, Any], BaseChatModel]] = None


def get_rag_components() -> Tuple[Dict[str, Any], BaseChatModel]:
    """Dependency injection function to provide RAG components."""
    global _GLOBAL_RAG_COMPONENTS
    if _GLOBAL_RAG_COMPONENTS is None:
        logger.info("Initializing RAG Components for the first time...")
        try:
            _GLOBAL_RAG_COMPONENTS = setup_rag()
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize backend models.")
    return _GLOBAL_RAG_COMPONENTS


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    rag_backend: Tuple[Dict[str, Any], BaseChatModel] = Depends(get_rag_components)
) -> ChatResponse:
    """
    Accepts a query, retrieves relevant documents, reranks them, and generates an answer.
    """
    components, llm = rag_backend
    query = request.query

    logger.info(f"Received API query: '{query}'")

    # 1. Agentic Retrieval
    try:
        docs = dynamic_retrieve(query, components, llm)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Document retrieval failed.")

    # 2. Mathematical Reranking
    try:
        docs = cohere_rerank(docs, query)
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(status_code=500, detail="Document reranking failed.")

    if not docs:
        return ChatResponse(
            answer="I could not find any relevant information in the mathematical PDFs to answer this question.",
            sources=[],
        )

    # 3. Combine Context
    context_text = "\n\n---\n\n".join(
        [f"[Source: {doc.metadata.get('source', 'Unknown Document')}]\n{doc.page_content}" for doc in docs]
    )

    formatted_prompt = prompt_template.format(context=context_text, question=query)

    # 4. LLM Generation
    try:
        response = llm.invoke(formatted_prompt)
    except Exception as e:
        logger.error(f"LLM Generation failed: {e}")
        raise HTTPException(status_code=500, detail="LLM generation failed.")

    # Handle Gemini vs Groq list-type response quirks
    content = response.content
    if isinstance(content, list):
        answer = " ".join([b.get("text", "") for b in content if isinstance(b, dict)]).strip()
    else:
        answer = str(content).strip()

    # 5. Format sources
    sources = [
        SourceDoc(source=doc.metadata.get("source", "Unknown Document"), content=doc.page_content)
        for doc in docs
    ]

    return ChatResponse(answer=answer, sources=sources)
