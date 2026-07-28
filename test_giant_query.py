import warnings
warnings.filterwarnings("ignore")

import os
from core import dynamic_retrieve, cohere_rerank, QDRANT_DB_DIR, STORE_DIR
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from storage import PickleFileStore
import pickle

def main():
    print("Loading Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Connecting to DB...")
    client = QdrantClient(path=QDRANT_DB_DIR)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="parent_document_store",
        embedding=embeddings,
    )
    store = PickleFileStore(STORE_DIR)
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    
    import pickle
    print("Loading BM25 Keyword Search Index...")
    with open("bm25_index.pkl", "rb") as f:
        bm25_retriever = pickle.load(f)

    components = {
        "vectorstore": vectorstore,
        "store": store,
        "parent_splitter": parent_splitter,
        "child_splitter": child_splitter,
        "bm25_retriever": bm25_retriever
    }

    from langchain_groq import ChatGroq
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    # 1. The original massive prompt (461 words)
    fluff = "I have been thinking a lot about the technology sector and analyzing various market trends. " * 30
    giant_question = fluff + "Anyway, what is the latest news on Apple's Q3 2023 financials?"
    
    print(f"\n--- Testing Giant Query ---")
    print(f"Total Words in Query: {len(giant_question.split())}")
    
    print("\n1. Running Dynamic Retrieval...")
    # This will hit our 256-token limit for vector search!
    retrieved_docs = dynamic_retrieve(giant_question, components, llm)
    
    print(f"\nRetrieved {len(retrieved_docs)} docs before reranking:")
    for i, doc in enumerate(retrieved_docs):
        print(f"  [{i}] {doc.metadata.get('source', 'Unknown')} | {doc.page_content[:50]}...")
        
    print("\n2. Running Cohere API Reranker...")
    reranked_docs = cohere_rerank(retrieved_docs, giant_question)
    
    print(f"\nReranked to {len(reranked_docs)} highly relevant docs:")
    for i, doc in enumerate(reranked_docs):
         print(f"  [{i}] {doc.metadata.get('source')} | {doc.page_content[:50]}...")

if __name__ == "__main__":
    main()
