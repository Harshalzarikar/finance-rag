import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

# Import RAG components from our backend
from core import cohere_rerank, dynamic_retrieve, setup_rag
from semantic_cache import semantic_cache

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

# Add CORS middleware to allow React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (update to Vercel URL in prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a highly intelligent, professional AI assistant specializing in Quantitative Finance and Mathematical Physics.\n\n"
        "Your rules:\n"
        "1. If the user is just saying a conversational greeting (like 'hi', 'hello', 'how are you'), respond politely and ask how you can help them with their research. Ignore the math context completely.\n"
        "2. For all other queries, answer the question using ONLY the provided context.\n"
        "3. If the answer is not contained within the context, you MUST say 'I don't know based on the provided documents.' Do not invent math problems or hallucinate dialogues."
    )),
    ("user", "Context:\n{context}\n\nQuestion: {question}")
])


# Pydantic models for request and response validation
class ChatRequest(BaseModel):
    query: str


class SourceDoc(BaseModel):
    source: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    cached: bool = False          # True when response came from semantic cache


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

    # 0. Short-circuit greetings BEFORE they hit the retrieval pipeline.
    #    Without this, "hi" gets searched in the vector DB, matches math
    #    subscripts like h_i, and the LLM hallucinates on garbage context.
    GREETINGS = {"hi", "hello", "hey", "hii", "hiii", "sup", "yo", "hola",
                 "good morning", "good afternoon", "good evening",
                 "how are you", "what's up", "whats up"}
    if query.strip().lower().rstrip("!?.") in GREETINGS:
        logger.info("Detected greeting — skipping retrieval pipeline.")
        return ChatResponse(
            answer="Hello! I'm your Quantitative Finance AI assistant. "
                   "Ask me anything about stochastic volatility, options pricing, "
                   "portfolio optimization, risk models, or any topic from the ArXiv research papers.",
            sources=[],
        )

    # 1. Semantic Cache Check — skip retrieval + LLM if a similar query was cached.
    cached_entry = semantic_cache.get(query)
    if cached_entry is not None:
        return ChatResponse(
            answer=cached_entry.answer,
            sources=cached_entry.sources,
            cached=True,
        )
    # 2. Agentic Retrieval
    try:
        docs = dynamic_retrieve(query, components, llm)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Document retrieval failed.")

    # 3. Mathematical Reranking
    try:
        docs = cohere_rerank(docs, query)
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(status_code=500, detail="Document reranking failed.")

    # 4. Short-circuit — no relevant docs found
    if not docs:
        return ChatResponse(
            answer="I could not find any relevant information in the mathematical PDFs to answer this question.",
            sources=[],
        )

    # 5. Combine Context
    context_text = "\n\n---\n\n".join(
        [f"[Source: {doc.metadata.get('source', 'Unknown Document')}]\n{doc.page_content}" for doc in docs]
    )

    messages = prompt_template.format_messages(context=context_text, question=query)

    # 6. LLM Generation
    try:
        response = llm.invoke(messages)
    except Exception as e:
        logger.error(f"LLM Generation failed: {e}")
        raise HTTPException(status_code=500, detail="LLM generation failed.")

    # Handle Gemini vs Groq list-type response quirks
    content = response.content
    if isinstance(content, list):
        answer = " ".join([b.get("text", "") for b in content if isinstance(b, dict)]).strip()
    else:
        answer = str(content).strip()

    # 7. Format sources
    sources = [
        SourceDoc(source=doc.metadata.get("source", "Unknown Document"), content=doc.page_content)
        for doc in docs
    ]

    # 8. Store in semantic cache for future similar queries
    semantic_cache.set(query, answer, sources)

    return ChatResponse(answer=answer, sources=sources, cached=False)


@app.get("/cache/stats")
async def cache_stats() -> dict:
    """Returns semantic cache hit/miss statistics."""
    return semantic_cache.stats()


@app.get("/cache/clear")
async def cache_clear() -> dict:
    """Clears all semantic cache entries."""
    semantic_cache.clear()
    return {"status": "cache cleared"}
