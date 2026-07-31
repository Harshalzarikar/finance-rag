# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for PyMuPDF and C extensions)
RUN apt-get update && apt-get install -y gcc g++ build-essential

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Render uses PORT env variable, HF Spaces defaults to 7860
ENV PORT=7860
EXPOSE ${PORT}

# Download the heavy index files, then start the server
CMD python download_indexes.py && uvicorn api:app --host 0.0.0.0 --port ${PORT}
