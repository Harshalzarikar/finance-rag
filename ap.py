import os
import pickle
from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_classic.storage import LocalFileStore
from langchain_core.stores import BaseStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found in .env")
    exit(1)

# ---------------------------------------------------------------------------
# Configuration (Must exactly match retriever_setup.py)
# ---------------------------------------------------------------------------
QDRANT_DB_DIR = "./qdrant_db_local"
STORE_DIR = "./doc_store_local"

class PickleFileStore(BaseStore):
    """Wraps LocalFileStore to serialize Document objects via pickle."""
    def __init__(self, path):
        self._store = LocalFileStore(path)
    def mget(self, keys):
        raw_values = self._store.mget(keys)
        return [pickle.loads(v) if v is not None else None for v in raw_values]
    def mset(self, key_value_pairs):
        self._store.mset([(k, pickle.dumps(v)) for k, v in key_value_pairs])
    def mdelete(self, keys):
        self._store.mdelete(keys)
    def yield_keys(self, prefix=None):
        yield from self._store.yield_keys(prefix=prefix)

def setup_rag():
    print("Loading Local HuggingFace Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Connecting to Qdrant Database...")
    client = QdrantClient(path=QDRANT_DB_DIR)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="parent_document_store",
        embedding=embeddings,
    )

    print("Connecting to Local Document Store...")
    store = PickleFileStore(STORE_DIR)
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

    print("Initializing Groq AI...")
    import os
    if os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.3-70b-versatile")
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
    
    print("Loading BM25 Keyword Search Index...")
    try:
        from langchain_community.retrievers import BM25Retriever
        with open("bm25_index.pkl", "rb") as f:
            bm25_retriever = pickle.load(f)
    except Exception as e:
        print(f"\n⚠️ Warning: Could not load BM25 index ({e}).")
        bm25_retriever = None

    components = {
        "vectorstore": vectorstore,
        "store": store,
        "parent_splitter": parent_splitter,
        "child_splitter": child_splitter,
        "bm25_retriever": bm25_retriever
    }
    return components, llm

def extract_company_filter(query, llm):
    prompt = PromptTemplate.from_template(
        "You are a query analyzer. Extract the exact name of the company the user is asking about.\n"
        "If they are asking a general question (e.g. 'any companies', 'escaped prisoner'), return EXACTLY the word 'NONE'.\n"
        "Query: {query}\n"
        "Company Name:"
    )
    try:
        response = llm.invoke(prompt.format(query=query)).content.strip()
        import re
        response = re.sub(r'["\']', '', response)
        if response.upper() == "NONE": return None
        return response
    except:
        return None

def compress_query(query, llm):
    """Uses LLM to summarize a long, noisy user prompt into a short search query."""
    if len(query.split()) < 30:
        return query
        
    prompt = PromptTemplate.from_template(
        "You are an expert search engine query extractor.\n"
        "Extract the core question or intent from the user's text into a single, concise search sentence.\n"
        "Do NOT answer the question. ONLY output the compressed search query.\n\n"
        "User Text: {text}\n"
        "Compressed Query:"
    )
    from langchain_core.output_parsers import StrOutputParser
    chain = prompt | llm | StrOutputParser()
    try:
        compressed = chain.invoke({"text": query}).strip()
        print(f"  [Query Compressor] Compressed '{len(query.split())} words' down to '{len(compressed.split())} words' for Vector Search.")
        return compressed
    except:
        return query

def dynamic_retrieve(query, components, llm):
    company = extract_company_filter(query, llm)
    search_kwargs = {}
    if company:
        from qdrant_client.http import models as rest
        search_kwargs = {
            "filter": rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.company",
                        match=rest.MatchText(text=company),
                    )
                ]
            )
        }
        print(f"  [Router] Autonomously applying Qdrant filter for: {company}")
        
    pr = ParentDocumentRetriever(
        vectorstore=components["vectorstore"],
        docstore=components["store"],
        child_splitter=components["child_splitter"],
        parent_splitter=components["parent_splitter"],
        search_kwargs=search_kwargs
    )
    
    if components["bm25_retriever"]:
        from langchain_classic.retrievers import EnsembleRetriever
        final_retriever = EnsembleRetriever(retrievers=[components["bm25_retriever"], pr], weights=[0.4, 0.6])
    else:
        final_retriever = pr
        
    # Compress the query before passing to Vector Database to avoid 256-token truncation
    search_query = compress_query(query, llm)
    
    return final_retriever.invoke(search_query)

def llm_rerank(docs, query, fallback_llm):
    """Uses Groq (Llama3) as a lightning fast cross-encoder to rerank/filter documents in ONE API call."""
    if not docs: return []
    
    import os
    try:
        from langchain_groq import ChatGroq
        eval_llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        print(f"  [Reranker Warning] Groq not configured properly, falling back to Gemini... ({e})")
        eval_llm = fallback_llm
        
    # Combine all docs into a numbered list
    docs_text = ""
    for i, doc in enumerate(docs):
        docs_text += f"\n[Document ID: {i}]\n{doc.page_content}\n"
    
    rerank_prompt = PromptTemplate.from_template(
        "You are a strict relevance judge.\n\n"
        "Query: {query}\n\n"
        "Documents:\n{documents}\n\n"
        "Which of the above documents contain information that helps answer the query?\n"
        "Return ONLY a comma-separated list of the relevant Document IDs (e.g. 0, 2, 3). If NONE of them are relevant, return 'NONE'."
    )
    
    print(f"  [Reranker] Evaluating {len(docs)} documents in a single fast call...")
    try:
        prompt_text = rerank_prompt.format(query=query, documents=docs_text)
        response = eval_llm.invoke(prompt_text)
        result = response.content.strip()
        
        if "none" in result.lower():
            print("  [Reranker] Kept 0 highly relevant documents.")
            return []
            
        # Parse the IDs from the response
        import re
        kept_ids = [int(s) for s in re.findall(r'\d+', result)]
        reranked = [docs[i] for i in kept_ids if i < len(docs)]
        
        print(f"  [Reranker] Kept {len(reranked)} highly relevant documents.")
        return reranked
    except Exception as e:
        print(f"  [Reranker Error] {e} - Falling back to keeping all documents.")
        return docs

def main():
    components, llm = setup_rag()
    
    prompt_template = PromptTemplate.from_template(
        "You are an AI assistant answering questions based on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    
    print("\n" + "="*50)
    print("RAG System is Ready! Ask a question (or type 'quit' to exit)")
    print("="*50)
    
    while True:
        try:
            query = input("\nYour Question: ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        print("Searching database...")
        # Retrieve relevant documents dynamically!
        docs = dynamic_retrieve(query, components, llm)
        
        if not docs:
            print("No relevant information found in the database.")
            continue
            
        # --- NEW RERANKING STEP ---
        docs = llm_rerank(docs, query, llm)
        
        if not docs:
            print("  [X] The AI determined that none of the retrieved documents answer the question. Please try rephrasing.")
            continue
            
        # Combine the text of all retrieved documents, injecting metadata!
        context_text = "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown Document')}]\n{doc.page_content}" 
            for doc in docs
        ])
        
        print("Asking Groq...")
        # Format the prompt and ask LLM
        formatted_prompt = prompt_template.format(context=context_text, question=query)
        
        try:
            response = llm.invoke(formatted_prompt)
        except Exception as e:
            print(f"\n⚠️ [API Error]: {e}")
            continue
        
        print("\n" + "-"*50)
        print("Answer:\n")
        
        # In newer LangChain versions, Gemini sometimes returns a list of blocks
        content = response.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and 'text' in block:
                    print(block['text'])
                else:
                    print(str(block))
        else:
            print(content)
            
        print("-" * 50)
        
        print("\nSources Used:")
        for i, doc in enumerate(docs):
            source_file = doc.metadata.get('source', 'Unknown Document')
            print(f"  [{i+1}] {source_file}")
            
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()