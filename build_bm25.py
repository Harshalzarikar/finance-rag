import logging
import os
import pickle

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

DOCS_DIR = "./real_pdfs"
BM25_FILE = "bm25_index.pkl"


def main() -> None:
    """
    Loads all documents lazily and constructs a local BM25 keyword index 
    for ensemble retrieval.
    """
    logger.info(f"Loading documents from {DOCS_DIR}...")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        use_multithreading=True,
    )

    logger.info("Lazily loading and splitting into Parent chunks...")
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    
    parent_docs = []
    # Use lazy_load to avoid OOM errors
    for doc in tqdm(loader.lazy_load(), desc="Streaming PDFs for BM25"):
        # Split immediately and append
        splits = parent_splitter.split_documents([doc])
        parent_docs.extend(splits)

    logger.info(f"Total parent chunks to index: {len(parent_docs)}")

    logger.info("Building BM25 Index (This will take a few seconds)...")
    bm25_retriever = BM25Retriever.from_documents(parent_docs)
    bm25_retriever.k = 4  # return top 4

    logger.info(f"Saving BM25 index to {BM25_FILE}...")
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25_retriever, f)

    logger.info("BM25 Index successfully created!")


if __name__ == "__main__":
    main()
