import logging
import os
import time

from dotenv import load_dotenv
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm

# Import the shared PickleFileStore from our main backend module
from core import PickleFileStore

# ---------------------------------------------------------------------------
# Setup Logging & Environment
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOCS_DIR = "./real_pdfs"
QDRANT_DB_DIR = "./qdrant_db_local"
STORE_DIR = "./doc_store_local"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors


def main() -> None:
    """
    Initializes the document loader, creates the Qdrant vector store and local 
    file store, and streams documents in batches for memory-safe indexing.
    """
    logger.info("Initializing Document Loader...")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        use_multithreading=True,
    )
    # We will use .lazy_load() later to stream documents one-by-one!

    logger.info("Initializing Local HuggingFace Embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Model loaded successfully!")

    logger.info("Initializing Qdrant Vector Store...")
    client = QdrantClient(path=QDRANT_DB_DIR)

    if not client.collection_exists("parent_document_store"):
        logger.info("Creating new Qdrant collection...")
        client.create_collection(
            collection_name="parent_document_store",
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="parent_document_store",
        embedding=embeddings,
    )

    logger.info("Initializing Local File Store...")
    os.makedirs(STORE_DIR, exist_ok=True)
    store = PickleFileStore(STORE_DIR)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    logger.info("Setting up ParentDocumentRetriever...")
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # -----------------------------------------------------------------------
    # Index documents in TRUE lazy batches to prevent RAM spikes!
    # -----------------------------------------------------------------------
    BATCH_SIZE = 20
    logger.info(f"Lazily streaming and indexing documents in batches of {BATCH_SIZE}...")
    start_time = time.time()

    batch = []

    # lazy_load() yields one document at a time directly from the hard drive,
    # preventing massive RAM spikes!
    for doc in tqdm(loader.lazy_load(), desc="Streaming & Indexing PDFs"):
        # Add metadata on the fly
        first_line = doc.page_content.split('\n')[0].strip()
        doc.metadata["company"] = first_line[:50] if first_line else "ArXiv Paper"

        batch.append(doc)

        # When batch is full, process it and clear it from RAM
        if len(batch) >= BATCH_SIZE:
            retriever.add_documents(batch)
            batch.clear()  # FREES RAM!

    # Process any remaining documents in the final partial batch
    if batch:
        retriever.add_documents(batch)
        batch.clear()

    total_time = time.time() - start_time
    logger.info(f"Indexing Complete! Total time: {total_time:.1f} seconds ({total_time / 60:.2f} minutes)")

    # -----------------------------------------------------------------------
    # Test Query
    # -----------------------------------------------------------------------
    logger.info("Running a test query...")
    test_query = "What is the main topic of the first document?"
    retrieved_docs = retriever.invoke(test_query)

    logger.info(f"Retrieved {len(retrieved_docs)} parent documents for the query: '{test_query}'")
    if retrieved_docs:
        logger.info("Snippet of the first retrieved parent document:")
        logger.info("-" * 50)
        logger.info(retrieved_docs[0].page_content[:500] + "...")
        logger.info("-" * 50)
    else:
        logger.warning("No documents retrieved.")


if __name__ == "__main__":
    main()