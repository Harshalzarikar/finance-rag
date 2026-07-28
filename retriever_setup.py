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

from storage import PickleFileStore

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
    )


    logger.info("Initializing Local HuggingFace Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Initializing Qdrant Vector Store...")
    client = QdrantClient(path=QDRANT_DB_DIR)

    if client.collection_exists("parent_document_store"):
        col_info = client.get_collection("parent_document_store")
        if col_info.config.params.vectors.size != EMBEDDING_DIM:
            logger.warning("Dimension mismatch detected. Recreating Qdrant collection...")
            client.delete_collection("parent_document_store")
            client.create_collection(
                collection_name="parent_document_store",
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
    else:
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
    # Index documents in lazy batches
    # -----------------------------------------------------------------------
    BATCH_SIZE = 20
    logger.info(f"Streaming and indexing documents in batches of {BATCH_SIZE}...")
    start_time = time.time()

    batch = []

    for doc in tqdm(loader.lazy_load(), desc="Streaming & Indexing PDFs"):
        batch.append(doc)

        # When batch is full, process it and clear it from RAM
        if len(batch) >= BATCH_SIZE:
            retriever.add_documents(batch)
            batch.clear()

    # Process any remaining documents in the final partial batch
    if batch:
        retriever.add_documents(batch)
        batch.clear()

    total_time = time.time() - start_time
    logger.info(f"Indexing Complete! Total time: {total_time:.1f} seconds ({total_time / 60:.2f} minutes)")


if __name__ == "__main__":
    main()