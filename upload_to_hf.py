import os
from huggingface_hub import HfApi

class HuggingFaceUploader:
    def __init__(self, repo_id: str):
        self.api = HfApi()
        self.repo_id = repo_id
        self.repo_type = "dataset"
        
        # Automatically create the dataset repository if it doesn't exist yet
        try:
            print(f"Ensuring repository '{self.repo_id}' exists on Hugging Face...")
            self.api.create_repo(repo_id=self.repo_id, repo_type=self.repo_type, exist_ok=True)
        except Exception as e:
            print(f"Note: Could not auto-create repo (it might already exist). Details: {e}")

    def upload_item(self, local_path: str):
        """Uploads a file or a folder to the Hugging Face Hub."""
        if not os.path.exists(local_path):
            print(f"Warning: '{local_path}' not found locally. Skipping.")
            return

        print(f"Uploading {local_path} (this may take a while depending on file size)...")
        
        if os.path.isdir(local_path):
            self.api.upload_folder(
                folder_path=local_path,
                path_in_repo=local_path,
                repo_id=self.repo_id,
                repo_type=self.repo_type
            )
        else:
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=local_path,
                repo_id=self.repo_id,
                repo_type=self.repo_type
            )
            
        print(f"Successfully uploaded {local_path}!")

if __name__ == "__main__":
    import shutil
    
    # The repository where your databases will be stored
    REPO_ID = "zarikarharry1412/finance-rag-indexes"
    
    print(f"Starting upload process to: {REPO_ID}")
    
    uploader = HuggingFaceUploader(repo_id=REPO_ID)
    
    # qdrant already uploaded, but keeping it here just in case
    uploader.upload_item("qdrant_db_local")
    
    # Zip doc_store_local because it contains thousands of tiny files
    # which makes Hugging Face's API very slow if uploaded one by one
    print("Zipping doc_store_local into doc_store_local.zip to speed up upload...")
    if os.path.exists("doc_store_local"):
        shutil.make_archive("doc_store_local", "zip", "doc_store_local")
        uploader.upload_item("doc_store_local.zip")
    
    uploader.upload_item("bm25_index.pkl")
    
    print("\nAll uploads completed! Your data is now in the cloud.")
