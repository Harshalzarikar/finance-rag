import os
import pickle
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

DOCS_DIR = "./documents"
BM25_FILE = "bm25_index.pkl"

def main():
    print(f"Loading documents from {DOCS_DIR}...")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        use_multithreading=True,
        show_progress=True,
    )
    docs = loader.load()
    
    print("Splitting into Parent chunks (matching semantic DB)...")
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
    parent_docs = parent_splitter.split_documents(docs)
    
    print(f"Total parent chunks to index: {len(parent_docs)}")
    
    print("Building BM25 Index (This will take a few seconds)...")
    bm25_retriever = BM25Retriever.from_documents(parent_docs)
    bm25_retriever.k = 4 # return top 4
    
    print(f"Saving BM25 index to {BM25_FILE}...")
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    print("BM25 Index successfully created!")

if __name__ == "__main__":
    main()
