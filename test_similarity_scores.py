import warnings
warnings.filterwarnings("ignore")

from core import QDRANT_DB_DIR
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    print("Loading Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Connecting to DB...")
    client = QdrantClient(path=QDRANT_DB_DIR)
    
    from langchain_qdrant import QdrantVectorStore
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="parent_document_store",
        embedding=embeddings,
    )
    
    # 1. Create the giant query
    fluff = "I have been thinking a lot about the technology sector and analyzing various market trends. " * 80
    giant_question = fluff+"Anyway, what is the latest news on Apple's Q3 2023 financials?"
    
    print("\nEmbedding the giant question (over 450 words)...")
    
    # 3. Test 1: Pure Vector Similarity Search (NO FILTERS)
    print("\n--- TEST 1: Pure Vector Search (No Filters) ---")
    print("Let's see what Qdrant thinks is most similar to the truncated fluff vector:")
    results_no_filter = vectorstore.similarity_search_with_score(giant_question, k=3)
    
    for i, (doc, score) in enumerate(results_no_filter):
        company = doc.metadata.get("company", "Unknown")
        print(f"  [{i}] Company: {company} | Similarity Score: {score:.4f}")
        
    # 4. Test 2: Vector Search WITH the LLM Router's Filter
    print("\n--- TEST 2: Vector Search WITH the LLM 'Apple' Filter ---")
    print("The Router forces Qdrant to only look at Apple documents.")
    
    # In Langchain Qdrant, we can pass metadata filters via kwargs
    # We will use the REST filter format required by Langchain Qdrant
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    apple_filter = Filter(must=[FieldCondition(key="metadata.company", match=MatchValue(value="Apple"))])
    
    results_with_filter = vectorstore.similarity_search_with_score(giant_question, k=3, filter=apple_filter)
    
    for i, (doc, score) in enumerate(results_with_filter):
        company = doc.metadata.get("company", "Unknown")
        print(f"  [{i}] Company: {company} | Similarity Score: {score:.4f}")

    # 5. Test 3: Vector Search WITH Agentic Query Compression
    print("\n--- TEST 3: Pure Vector Search WITH Agentic Query Compression ---")
    print("Let's compress the query first using the new ap.py architecture...")
    from core import compress_query
    from langchain_groq import ChatGroq
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    compressed_query = compress_query(giant_question, llm)
    print(f"Compressed Query: {compressed_query}")
    
    compressed_vector = embeddings.embed_query(compressed_query)
    results_compressed = vectorstore.similarity_search_with_score(compressed_query, k=3)
    
    print("\nLet's see what Qdrant thinks is most similar to the COMPRESSED vector (No Filters!):")
    for i, (doc, score) in enumerate(results_compressed):
        company = doc.metadata.get("company", "Unknown")
        print(f"  [{i}] Company: {company} | Similarity Score: {score:.4f}")

if __name__ == "__main__":
    main()
