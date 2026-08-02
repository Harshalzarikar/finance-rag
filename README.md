
# 📈 Quantitative Finance RAG

> A production-grade, locally-validated Retrieval-Augmented Generation (RAG) system built over 1,190 ArXiv quantitative finance research papers (~31,000 pages, ~20M tokens).

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **LLM** | Llama 3.3 70B via Groq API (low-latency inference) |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace, CPU-friendly, 384-dim) |
| **Vector DB** | Qdrant (local HNSW index, ~112,000 vectors) |
| **Keyword Search** | BM25 (rank-bm25) |
| **Reranker** | Cohere Rerank v3.0 (cross-encoder) |
| **Backend** | FastAPI + Uvicorn + Pydantic v2 |
| **Frontend** | React + Vite |
| **Evaluation** | RAGAS Framework (LLM-as-a-Judge) |
| **PDF Extraction** | PyMuPDF (C++ bindings, fastest available) |

---

## 📊 RAGAS Evaluation Results (LLM-as-a-Judge)

Evaluated using the **RAGAS Framework** against a constrained Q&A test set over quantitative finance documents:

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Context Precision** | **1.0** | Retrieved chunks are relevant to the query |
| **Context Recall** | **1.0** | All necessary context is retrieved |
| **Faithfulness** | **1.0** | Answers are grounded in source documents (no hallucination) |

> Results exported to [`ragas_evaluation_results.csv`](./ragas_evaluation_results.csv). Scores reflect performance on the constrained test set — see [`SYSTEM_ARCHITECTURE.md`](./SYSTEM_ARCHITECTURE.md) for full evaluation methodology.

---

## 🌟 Architecture Overview

```
User Query
    │
    ▼
Query Compressor (LLM)          → condenses long queries into precise search terms
    │
    ▼
Ensemble Retriever
    ├── Qdrant Vector Search  (weight: 0.6)  → semantic meaning
    └── BM25 Keyword Search   (weight: 0.4)  → exact term matching
    │
    ▼  Reciprocal Rank Fusion (RRF)
    │
    ▼
Cohere Rerank v3.0              → cross-encoder scores candidates, keeps top 2
    │
    ▼
Groq Llama 3.3 70B              → synthesizes final answer with full parent-chunk context
    │
    ▼
FastAPI Response                → answer + source citations (PDF filename + excerpt)
```

### Key Design Decisions

1. **Parent-Child Chunking** — Child chunks (250 tokens) are embedded for precise semantic search. On retrieval, they map back to their Parent chunk (1,000 tokens) fed to the LLM — preventing the "Lost in the Middle" problem in dense mathematical text.

2. **Hybrid BM25 + Qdrant Retrieval** — Pure semantic search fails on exact finance acronyms (e.g., `LTRO`, `e-MID`, `HJM`). BM25 covers exact keyword matching; Qdrant covers conceptual similarity. Both are fused via RRF.

3. **Memory-Safe Ingestion** — `PyMuPDF` with `lazy_load()` + `BATCH_SIZE=20` keeps peak RAM under **250 MB** while processing 31,000 pages — runnable on a standard laptop.

4. **Greeting Short-Circuit** — Greeting queries (`hi`, `hello`, etc.) are intercepted before they hit the retrieval pipeline, preventing the LLM from matching math subscripts like `h_i` to "hi" and hallucinating.

---

## 🧮 Local Hardware Footprint (Measured)

| Component | Actual Size | Description |
| :--- | :--- | :--- |
| **Raw PDFs (Input)** | `1.56 GB` | 1,190 ArXiv quantitative finance papers |
| **Qdrant Vector DB** | `579 MB` | HNSW index (`all-MiniLM-L6-v2`, 384-dim) |
| **BM25 Sparse Index** | `164 MB` | Serialized keyword matching dictionary |
| **Parent Doc Store** | `104 MB` | Pickled parent chunks (LLM context) |
| **Total Index Size** | **~847 MB** | Full retrieval system on disk |
| **Peak RAM (Ingestion)** | `< 250 MB` | Via `lazy_load()` batching |

---

## ⚠️ Known Limitation & Enterprise Path

**Concurrent Multi-Worker Access:**
The local Qdrant file-based client uses a file lock — only one process can open the database at a time. This means `uvicorn api:app --workers N` (N > 1) will crash all workers except the first.

**Current behaviour:** Single-worker async FastAPI handles concurrent users correctly.
**Enterprise fix:** Replace `QdrantClient(path=...)` with `QdrantClient(url=..., api_key=...)` pointing to a Qdrant Cloud / self-hosted server instance. This is a **2-line code change** — the rest of the architecture is identical.

> See [`SYSTEM_ARCHITECTURE.md`](./SYSTEM_ARCHITECTURE.md) for the full enterprise architecture design — including AWS S3, SQS, Kubernetes, and cost projections up to 10M documents.

---

## 🚀 Setup & Installation

### 1. Clone & Install

```bash
git clone https://github.com/Harshalzarikar/finance-rag.git
cd finance-rag
python -m venv venv
# Activate (Windows)
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY="your_groq_key"
COHERE_API_KEY="your_cohere_key"
GOOGLE_API_KEY="your_gemini_key"
HF_DATASET_REPO="zarikarharry1412/finance-rag-indexes"
```

---

## 🛠️ Usage Pipeline

> The raw PDFs and database files are excluded from the repo via `.gitignore`. You must generate them locally following the steps below.

**Step 1 — Download the Dataset**
```bash
python download_pdfs.py
```
*Downloads 1,190 ArXiv quantitative finance PDFs into `./real_pdfs`*

**Step 2 — Build the Vector Database**
```bash
python retriever_setup.py
```
*Chunks 31,000+ pages into ~112,000 vectors and persists them to `./qdrant_db_local`. Takes ~1 hour on a standard CPU.*

**Step 3 — Build the Keyword Index**
```bash
python build_bm25.py
```
*Builds the BM25 sparse index and serializes it to `bm25_index.pkl`*

**Step 4 — Run the API Server**
```bash
uvicorn api:app --workers 1
```
*Starts the FastAPI server on `http://localhost:8000`. Use `--workers 1` — local Qdrant does not support multi-process concurrent access.*

**Step 5 — Evaluate**
```bash
python production_eval.py
```
*Runs RAGAS evaluation and exports scores to `ragas_evaluation_results.csv`*

---

## 📁 Project Structure

```
finance-rag/
├── api.py                  # FastAPI server (endpoints, dependency injection)
├── core.py                 # RAG pipeline (retrieval, reranking, generation)
├── retriever_setup.py      # PDF ingestion → Qdrant + doc store
├── build_bm25.py           # BM25 sparse index builder
├── storage.py              # Custom pickle-based document store
├── download_pdfs.py        # ArXiv dataset downloader
├── download_indexes.py     # HuggingFace index downloader (for deployment)
├── production_eval.py      # RAGAS evaluation runner
├── frontend/               # React + Vite chat UI
├── SYSTEM_ARCHITECTURE.md  # Enterprise architecture design proposal
└── RAG_Evaluation_Report.md
```

---

*For the full enterprise scaling design (AWS S3 → SQS → Kubernetes → Qdrant Cloud), cost projections, and security architecture, see [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md).*
