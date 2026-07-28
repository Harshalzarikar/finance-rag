import os
import json
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_classic.storage import LocalFileStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOCS_DIR = "./real_pdfs"
QDRANT_DB_DIR = "./qdrant_db_local"
STORE_DIR = "./doc_store_local"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors


def main():
    print("Initializing Document Loader...")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        use_multithreading=True,
    )
    # We will use .lazy_load() later to stream documents one-by-one!

    # -----------------------------------------------------------------------
    # Local Embeddings — NO API keys, NO rate limits, NO internet needed!
    # Model: all-MiniLM-L6-v2 (only 22MB, downloads once and cached forever)
    # -----------------------------------------------------------------------
    print("Initializing Local HuggingFace Embeddings (all-MiniLM-L6-v2)...")
    print("(First run will download 22MB model — takes ~5 seconds)")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Model loaded successfully!")

    print("Initializing Qdrant Vector Store...")
    client = QdrantClient(path=QDRANT_DB_DIR)

    if not client.collection_exists("parent_document_store"):
        print("Creating new Qdrant collection...")
        client.create_collection(
            collection_name="parent_document_store",
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="parent_document_store",
        embedding=embeddings,
    )

    print("Initializing Local File Store...")
    os.makedirs(STORE_DIR, exist_ok=True)

    # LocalFileStore only handles raw bytes, but ParentDocumentRetriever
    # needs to store Document objects. We wrap it with pickle serialization.
    import pickle
    from langchain_core.stores import BaseStore

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

    store = PickleFileStore(STORE_DIR)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

    print("Setting up ParentDocumentRetriever...")
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
    print(f"\nLazily streaming and indexing documents in batches of {BATCH_SIZE}...")
    start_time = time.time()

    batch = []
    
    # lazy_load() yields one document at a time directly from the hard drive, 
    # preventing the massive 11 GB RAM spike!
    for doc in tqdm(loader.lazy_load(), desc="Streaming & Indexing PDFs"):
        # Add metadata on the fly
        first_line = doc.page_content.split('\n')[0].strip()
        doc.metadata["company"] = first_line[:50] if first_line else "ArXiv Paper"
        
        batch.append(doc)
        
        # When batch is full, process it and clear it from RAM
        if len(batch) >= BATCH_SIZE:
            retriever.add_documents(batch)
            batch.clear() # FREES RAM!

    # Process any remaining documents in the final partial batch
    if batch:
        retriever.add_documents(batch)
        batch.clear()

    total_time = time.time() - start_time
    print(f"\nIndexing Complete! Total time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
    # -----------------------------------------------------------------------
    # Test Query
    # -----------------------------------------------------------------------
    print("\nRunning a test query...")
    test_query = "What is the main topic of the first document?"
    retrieved_docs = retriever.invoke(test_query)

    print(f"\nRetrieved {len(retrieved_docs)} parent documents for the query: '{test_query}'")
    if retrieved_docs:
        print("\nSnippet of the first retrieved parent document:")
        print("-" * 50)
        print(retrieved_docs[0].page_content[:500] + "...")
        print("-" * 50)
    else:
        print("No documents retrieved.")


if __name__ == "__main__":
    main()