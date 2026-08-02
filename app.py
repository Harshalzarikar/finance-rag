import logging
import uvicorn

# 1. Download indexes from HF Datasets before loading the RAG core
import download_indexes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Downloading indexes from Hugging Face before starting server...")
    download_indexes.download_data()
except Exception as e:
    logger.warning(f"Failed to download indexes: {e}. Continuing with existing data if available.")

# 2. Import the FastAPI app (which will load the downloaded files on first request)
from api import app  # noqa: F401 - imported for uvicorn

if __name__ == "__main__":
    # Hugging Face Spaces (Docker SDK) uses port 7860
    uvicorn.run("api:app", host="0.0.0.0", port=7860)
