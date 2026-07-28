# Quantitative Finance RAG

A local Retrieval-Augmented Generation (RAG) pipeline built to ingest, vectorize, and reason over academic research papers in Quantitative Finance.

This system is capable of running completely locally (CPU-friendly ingestion) while leveraging state-of-the-art AI routing, reranking, and compression techniques to achieve maximum accuracy on 30,000+ pages of dense mathematical text.

## 🌟 Key Architecture Features

1. **FastAPI Server (`api.py`)**
   - REST API implemented with FastAPI.
   - Utilizes `Depends()` for dependency injection to manage retrievers.
   - Pydantic models for strict I/O validation.

2. **High-Speed Memory-Safe Ingestion (`retriever_setup.py`)**
   - Uses `PyMuPDF` (C++) for the fastest, most accurate PDF text extraction available in Python.
   - Implements strict `lazy_load()` Streaming and Batching Protocols (`BATCH_SIZE=20`) to safely process tens of thousands of pages on standard hardware without RAM explosions (Footprint capped at < 250 MB).
   - Persists data to a **Local Qdrant Vector Database** using `all-MiniLM-L6-v2` embeddings.
   - Uses `ParentDocumentRetriever` with explicit context preservation (`chunk_overlap=400` / `100`).

3. **Hybrid Ensemble Retrieval (`build_bm25.py`)**
   - Builds a high-speed BM25 Keyword Search dictionary.
   - Combines Semantic Vector Search (Qdrant) with Keyword Search (BM25) via a weighted `EnsembleRetriever`.

4. **Query Compression & Reranking (`core.py`)**
   - **Compressor:** Uses a lightweight LLM to summarize long user prompts into semantic search queries to avoid token-truncation.
   - **Cohere Reranker:** Passes the candidate documents through a cross-encoder (Cohere Rerank API) to score and filter chunks.
   - **Llama 3 Generation:** Synthesizes the final highly technical answer using Groq's high-speed Llama 3 API (or Gemini fallback).

5. **LLM-as-a-Judge Evaluation (`production_eval.py`)**
   - Implements the **Ragas Framework** to run automated evaluations against a constrained test set.
   - Automatically scores `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall`.
   - Includes automatic API throttling (`RunConfig`) and exports results to a Pandas DataFrame/CSV for CI/CD pipelines.

6. **Strict Software Engineering Standards**
   - Fully PEP-8 compliant.
   - Python `typing` library utilized across all functions.
   - Professional timestamped output via Python's standard `logging` module.

## 🧮 System Hardware Footprint (For 1,190 PDFs)

This system is engineered for maximum hardware efficiency. The following table represents the exact, live footprint required to index 1,190 quantitative finance PDFs (~31,000 pages):

| Component | Size | Description |
| :--- | :--- | :--- |
| **Raw PDFs (Input)** | `1.56 GB` | Total disk space of the 1,190 academic papers. |
| **Qdrant Vector DB** | `579 MB` | The HNSW semantic search index (`all-MiniLM-L6-v2` embeddings). |
| **Sparse Index** | `164 MB` | The serialized BM25 keyword matching dictionary. |
| **Local Doc Store** | `104 MB` | The pickled Parent chunks fed directly to the LLM. |
| **Peak RAM (Ingestion)** | `< 250 MB` | Local memory footprint kept low via `lazy_load()` batching (inference relies on cloud APIs). |

*(For a detailed mathematical breakdown on how this scales to 10 Million PDFs [20.1 TB], see [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md)).*

## 🚀 Setup & Installation

### 1. Clone & Install
```bash
git clone https://github.com/Harshalzarikar/finance-rag.git
cd finance-rag
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
*(Chunks 30,000+ pages and embeds ~80,000 vectors into `./qdrant_db_local`. This takes ~1 hour on a standard CPU).*

**Step 3: Build the Keyword Dictionary**
```bash
python build_bm25.py
```
*(Builds the exact keyword matching index).*

**Step 4: Run the API Server**
```bash
uvicorn api:app
```
*(Starts the production FastAPI server on `localhost:8000`)*

**Alternative: Run the Terminal Chat System**
```bash
python ap.py
```
*(Starts an interactive terminal where you can ask complex finance questions!)*

**Step 5: Evaluate the Architecture**
```bash
python production_eval.py
```
*(Runs the Ragas test cases and generates a `ragas_evaluation_results.csv` scorecard!)*
