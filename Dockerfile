# Use official Python 3.13 slim image
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Install build dependencies for PyMuPDF, Qdrant, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Copy the core application code
COPY ap.py .
COPY api.py .
COPY retriever_setup.py .
COPY build_bm25.py .

# Copy the built databases (Qdrant and BM25)
COPY qdrant_db_local ./qdrant_db_local
COPY bm25_index.pkl .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
