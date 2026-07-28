from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# Import RAG components from our backend
from ap import setup_rag, cohere_rerank, dynamic_retrieve

load_dotenv()

app = FastAPI(
    title="Quantitative Finance RAG API", 
    description="Production REST API for querying the Agentic RAG architecture."
)

# Initialize global components so we don't recreate the vector store on every request!
print("Initializing RAG Components on startup...")
try:
    components, llm = setup_rag()
except Exception as e:
    print(f"Failed to initialize RAG components: {e}")
    components, llm = None, None

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

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not components or not llm:
        raise HTTPException(status_code=500, detail="RAG Components not initialized properly. Check logs.")
        
    query = request.query
    
    # 1. Agentic Retrieval
    try:
        docs = dynamic_retrieve(query, components, llm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
        
    # 2. Mathematical Reranking
    try:
        docs = cohere_rerank(docs, query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {e}")
        
    if not docs:
        return ChatResponse(
            answer="I could not find any relevant information in the mathematical PDFs to answer this question.", 
            sources=[]
        )
        
    # 3. Combine Context
    context_text = "\n\n---\n\n".join([
        f"[Source: {doc.metadata.get('source', 'Unknown Document')}]\n{doc.page_content}" 
        for doc in docs
    ])
    
    formatted_prompt = prompt_template.format(context=context_text, question=query)
    
    # 4. LLM Generation
    try:
        response = llm.invoke(formatted_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {e}")
        
    # Handle Gemini vs Groq list-type response quirks
    content = response.content
    if isinstance(content, list):
        answer = " ".join([b.get("text", "") for b in content if isinstance(b, dict)]).strip()
    else:
        answer = content.strip()
        
    # 5. Format sources
    sources = [
        SourceDoc(source=doc.metadata.get('source', 'Unknown Document'), content=doc.page_content)
        for doc in docs
    ]
    
    return ChatResponse(answer=answer, sources=sources)
