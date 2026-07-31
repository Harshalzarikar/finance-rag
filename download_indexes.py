import os
import shutil
from huggingface_hub import snapshot_download

def download_data():
    """
    Downloads all local database assets from a Hugging Face Dataset.
    This runs once on container startup before the FastAPI server boots.
    
    Expected HF Dataset structure:
        ├── bm25_index.pkl
        ├── doc_store_local/
        │   └── ... (pickle files)
        └── qdrant_db_local/
            └── ... (qdrant collection files)
    """
    REPO_ID = os.environ.get("HF_DATASET_REPO", "Harshalzarikar/finance-rag-indexes")
    
    print(f"Downloading indexes from Hugging Face Dataset: {REPO_ID}")
    try:
        local_dir = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
        
        # 1. BM25 sparse index
        src = os.path.join(local_dir, "bm25_index.pkl")
        if os.path.exists(src):
            shutil.copy(src, "./bm25_index.pkl")
            print("Loaded bm25_index.pkl")
            
        # 2. Parent document store (now as a ZIP)
        src_zip = os.path.join(local_dir, "doc_store_local.zip")
        if os.path.exists(src_zip):
            print("Unzipping doc_store_local.zip...")
            if os.path.exists("./doc_store_local"):
                shutil.rmtree("./doc_store_local")
            shutil.unpack_archive(src_zip, "./doc_store_local")
            print("Loaded doc_store_local")

        # 3. Qdrant vector database
        src = os.path.join(local_dir, "qdrant_db_local")
        if os.path.isdir(src):
            if os.path.exists("./qdrant_db_local"):
                shutil.rmtree("./qdrant_db_local")
            shutil.copytree(src, "./qdrant_db_local")
            print("Loaded qdrant_db_local")
            
        print("Download complete. All databases ready.")
    except Exception as e:
        print(f"Warning: Failed to download from {REPO_ID}. "
              f"If running locally with existing data, you can ignore this. Error: {e}")

if __name__ == "__main__":
    download_data()
