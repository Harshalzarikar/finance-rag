import logging
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import cohere
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.stores import BaseStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

# ---------------------------------------------------------------------------
# Setup Logging & Environment
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    logger.error("GOOGLE_API_KEY not found in .env")
    raise ValueError("GOOGLE_API_KEY environment variable is missing.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QDRANT_DB_DIR = "./qdrant_db_local"
STORE_DIR = "./doc_store_local"


class PickleFileStore(BaseStore):
    """Wraps LocalFileStore to serialize Document objects via pickle for local storage."""

    def __init__(self, path: str) -> None:
        self._store = LocalFileStore(path)

    def mget(self, keys: List[str]) -> List[Optional[Document]]:
        raw_values = self._store.mget(keys)
        return [pickle.loads(v) if v is not None else None for v in raw_values]

    def mset(self, key_value_pairs: List[Tuple[str, Document]]) -> None:
        self._store.mset([(k, pickle.dumps(v)) for k, v in key_value_pairs])

    def mdelete(self, keys: List[str]) -> None:
        self._store.mdelete(keys)

    def yield_keys(self, prefix: Optional[str] = None) -> Any:
        yield from self._store.yield_keys(prefix=prefix)


def setup_rag() -> Tuple[Dict[str, Any], BaseChatModel]:
    """
    Initializes the embedding model, vector store, and language model.

    Returns:
        Tuple[Dict[str, Any], BaseChatModel]: A tuple containing a dictionary of RAG
        components (vectorstore, store, splitters, BM25) and the primary LLM instance.
    """
    logger.info("Loading Local HuggingFace Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Connecting to Qdrant Database...")
    client = QdrantClient(path=QDRANT_DB_DIR)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="parent_document_store",
        embedding=embeddings,
    )

    logger.info("Connecting to Local Document Store...")
    store = PickleFileStore(STORE_DIR)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

    logger.info("Initializing AI Models...")
    if os.environ.get("GROQ_API_KEY"):
        llm: BaseChatModel = ChatGroq(model="llama-3.3-70b-versatile")
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

    logger.info("Loading BM25 Keyword Search Index...")
    try:
        with open("bm25_index.pkl", "rb") as f:
            bm25_retriever = pickle.load(f)
    except Exception as e:
        logger.warning(f"Could not load BM25 index ({e}). Keyword search will be disabled.")
        bm25_retriever = None

    components = {
        "vectorstore": vectorstore,
        "store": store,
        "parent_splitter": parent_splitter,
        "child_splitter": child_splitter,
        "bm25_retriever": bm25_retriever,
    }
    return components, llm


def extract_company_filter(query: str, llm: BaseChatModel) -> Optional[str]:
    """
    Extracts the targeted company name from the user's query if present.

    Args:
        query (str): The raw user query.
        llm (BaseChatModel): The language model to use for extraction.

    Returns:
        Optional[str]: The company name, or None if no specific company is targeted.
    """
    prompt = PromptTemplate.from_template(
        "You are a query analyzer. Extract the exact name of the company the user is asking about.\n"
        "If they are asking a general question (e.g. 'any companies', 'escaped prisoner'), return EXACTLY the word 'NONE'.\n"
        "Query: {query}\n"
        "Company Name:"
    )
    try:
        response = llm.invoke(prompt.format(query=query)).content.strip()
        response = re.sub(r'["\']', '', str(response))
        if response.upper() == "NONE":
            return None
        return response
    except Exception as e:
        logger.error(f"Failed to extract company filter: {e}")
        return None


def compress_query(query: str, llm: BaseChatModel) -> str:
    """
    Summarizes a noisy or long user prompt into a concise search query.

    Args:
        query (str): The raw user query.
        llm (BaseChatModel): The language model.

    Returns:
        str: The compressed search query.
    """
    if len(query.split()) < 30:
        return query

    prompt = PromptTemplate.from_template(
        "You are an expert search engine query extractor.\n"
        "Extract the core question or intent from the user's text into a single, concise search sentence.\n"
        "Do NOT answer the question. ONLY output the compressed search query.\n\n"
        "User Text: {text}\n"
        "Compressed Query:"
    )
    chain = prompt | llm | StrOutputParser()
    try:
        compressed = chain.invoke({"text": query}).strip()
        logger.info(f"Compressed query from {len(query.split())} to {len(compressed.split())} words.")
        return compressed
    except Exception as e:
        logger.error(f"Failed to compress query: {e}")
        return query


def dynamic_retrieve(query: str, components: Dict[str, Any], llm: BaseChatModel) -> List[Document]:
    """
    Executes a hybrid search using both BM25 Sparse and Qdrant Dense vector retrieval.

    Args:
        query (str): The search query.
        components (Dict[str, Any]): The RAG components containing retrievers.
        llm (BaseChatModel): The language model.

    Returns:
        List[Document]: A list of relevant retrieved documents.
    """
    search_kwargs: Dict[str, Any] = {}

    pr = ParentDocumentRetriever(
        vectorstore=components["vectorstore"],
        docstore=components["store"],
        child_splitter=components["child_splitter"],
        parent_splitter=components["parent_splitter"],
        search_kwargs=search_kwargs,
    )

    if components["bm25_retriever"]:
        final_retriever = EnsembleRetriever(
            retrievers=[components["bm25_retriever"], pr], weights=[0.4, 0.6]
        )
    else:
        final_retriever = pr

    search_query = compress_query(query, llm)
    return final_retriever.invoke(search_query)


def cohere_rerank(docs: List[Document], query: str) -> List[Document]:
    """
    Uses the official Cohere Rerank API to mathematically score and filter documents.

    Args:
        docs (List[Document]): A list of candidate documents.
        query (str): The original search query.

    Returns:
        List[Document]: The filtered, highest-confidence documents (top 2).
    """
    if not docs:
        return []

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        logger.warning("COHERE_API_KEY not found in environment. Skipping reranking.")
        return docs

    try:
        co = cohere.Client(api_key)
        logger.info(f"Evaluating {len(docs)} documents via Cohere Reranker...")

        doc_texts = [doc.page_content for doc in docs]
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=doc_texts,
            top_n=2,
        )

        reranked_docs = [docs[res.index] for res in response.results]
        logger.info(f"Kept {len(reranked_docs)} highly relevant documents.")
        return reranked_docs
    except Exception as e:
        logger.error(f"Cohere API failed: {e}. Falling back to unranked documents.")
        return docs


def main() -> None:
    """Entry point for the terminal-based interactive chat."""
    components, llm = setup_rag()

    prompt_template = PromptTemplate.from_template(
        "You are an AI assistant answering questions based on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    print("\n" + "=" * 50)
    print("RAG System is Ready! Ask a question (or type 'quit' to exit)")
    print("=" * 50)

    while True:
        try:
            query = input("\nYour Question: ")
        except (KeyboardInterrupt, EOFError):
            break

        if query.lower() in ["quit", "exit", "q"]:
            break

        logger.info("Searching database...")
        docs = dynamic_retrieve(query, components, llm)

        if not docs:
            print("No relevant information found in the database.")
            continue

        docs = cohere_rerank(docs, query)

        if not docs:
            print("The AI determined that none of the retrieved documents answer the question.")
            continue

        context_text = "\n\n---\n\n".join(
            [f"[Source: {doc.metadata.get('source', 'Unknown Document')}]\n{doc.page_content}" for doc in docs]
        )

        logger.info("Generating response from LLM...")
        formatted_prompt = prompt_template.format(context=context_text, question=query)

        try:
            response = llm.invoke(formatted_prompt)
        except Exception as e:
            logger.error(f"API Error during generation: {e}")
            continue

        print("\n" + "-" * 50)
        print("Answer:\n")

        content = response.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    print(block["text"])
                else:
                    print(str(block))
        else:
            print(content)

        print("-" * 50)

        print("\nSources Used:")
        for i, doc in enumerate(docs):
            source_file = doc.metadata.get("source", "Unknown Document")
            print(f"  [{i + 1}] {source_file}")

        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()