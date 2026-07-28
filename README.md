# Quantitative Finance Agentic RAG

An Enterprise-grade Retrieval-Augmented Generation (RAG) architecture built to ingest, vectorize, and reason over massive datasets of highly complex academic research papers in Quantitative Finance.

This system is capable of running completely locally (CPU-friendly ingestion) while leveraging state-of-the-art AI routing, reranking, and compression techniques to achieve maximum accuracy on 30,000+ pages of dense mathematical text.

## 🌟 Key Architecture Features

1. **Massive Data Acquisition (`download_real_pdfs.py`)**
   - Autonomously scrapes the ArXiv API for thousands of Quantitative Finance (`q-fin`) research papers.
   - Saves real-world academic PDFs (charts, LaTeX math, data tables) locally for processing.

2. **High-Speed Memory-Safe Ingestion (`retriever_setup.py`)**
   - Uses `PyMuPDF` (C++) for the fastest, most accurate PDF text extraction available in Python.
   - Implements strict Batching Protocols (`BATCH_SIZE=20`) to safely process tens of thousands of pages on standard hardware without RAM explosions.
   - Persists data to a **Local Qdrant Vector Database** using `all-MiniLM-L6-v2` embeddings.

3. **Hybrid Ensemble Retrieval (`build_bm25.py`)**
   - Builds a high-speed BM25 Keyword Search dictionary.
   - Combines Semantic Vector Search (Qdrant) with Keyword Search (BM25) via a weighted `EnsembleRetriever`.

4. **Agentic Query Compression & Mathematical Reranking (`ap.py`)**
   - **Agentic Compressor:** Uses a lightweight LLM to compress noisy user questions into dense semantic search vectors to prevent token-truncation.
   - **Cohere Reranker:** Passes the candidate documents through the official `Cohere Rerank API` to mathematically filter out bad chunks before passing context to the final generation model.
   - **Llama 3 Generation:** Synthesizes the final highly technical answer using Groq's high-speed Llama 3 API (or Gemini fallback).

## 🚀 Setup & Installation

### 1. Clone & Install
```bash
git clone https://github.com/your-username/advance-rag.git
cd advance-rag
python -m venv venv
# Activate the virtual environment (Windows)
.\venv\Scripts\Activate.ps1 
# Install frozen dependencies
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
GOOGLE_API_KEY="your_gemini_key_here"
COHERE_API_KEY="your_cohere_key_here"
GROQ_API_KEY="your_groq_key_here"
```

## 🛠️ Usage Pipeline

> **Note:** The actual dataset (1.5GB of PDFs) and databases are blocked via `.gitignore` to keep this repository clean. You must generate them locally.

**Step 1: Download the Dataset**
```bash
python download_real_pdfs.py
```
*(Downloads 1,200 PDFs into `./real_pdfs`)*

**Step 2: Build the Vector Database**
```bash
python retriever_setup.py
```
*(Chunks 30,000+ pages and embeds ~230,000 vectors into `./qdrant_db_local`. This takes ~1 hour on a standard CPU).*

**Step 3: Build the Keyword Dictionary**
```bash
python build_bm25.py
```
*(Builds the exact keyword matching index).*

**Step 4: Run the RAG System**
```bash
python ap.py
```
*(Starts the interactive terminal where you can ask complex finance questions!)*
