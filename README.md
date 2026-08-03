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

## 🌟 Architecture Overview

```
User Query
    │
    ├──► Semantic Cache (cosine sim > 0.88) ──► Cache Hit ──► Fast FastAPI Response
    │
    ▼ Cache Miss
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
    │
    └──► Cache Set (stores query, answer, sources)
```

### Key Design Decisions

1. **Parent-Child Chunking** — Child chunks (700 chars / ~175 tokens) are embedded for precise semantic search. Size is capped at 700 chars specifically to stay within `all-MiniLM-L6-v2`'s hard 256-token context window — dense LaTeX can tokenize at 0.35 tokens/char. On retrieval, child chunks map back to their Parent chunk (4,000 chars / ~1,000 tokens) fed to the LLM for full context.

2. **Hybrid BM25 + Qdrant Retrieval** — Pure semantic search fails on exact finance acronyms (e.g., `LTRO`, `e-MID`, `HJM`). BM25 covers exact keyword matching; Qdrant covers conceptual similarity. Both are fused via RRF.

3. **Memory-Safe Ingestion** — `PyMuPDF` with `lazy_load()` + `BATCH_SIZE=20` keeps peak RAM under **250 MB** while processing 31,000 pages — runnable on a standard laptop.

4. **Semantic Caching** — Incoming queries are embedded and compared against past queries via cosine similarity. If similarity > 0.88, the API short-circuits and returns the cached answer instantly. Drops P99 latency from ~5-10s down to 0.01s for repeat financial queries, avoiding massive LLM API costs at scale.

5. **Greeting Short-Circuit** — Greeting queries (`hi`, `hello`, etc.) are intercepted before they hit the retrieval pipeline, preventing the LLM from matching math subscripts like `h_i` to "hi" and hallucinating.

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

## 🏭 Production Deployment Architecture

This project is a validated **Proof of Concept (POC)**. The table below shows the exact infrastructure swap required to take it to enterprise production — every component maps 1-to-1 with what's already built locally.

### Component Mapping: POC → Enterprise

| Component | Local POC (Built ✅) | Enterprise Production | Why the Change |
| :--- | :--- | :--- | :--- |
| **PDF Storage** | `./real_pdfs/` (local disk) | AWS S3 / GCS Bucket | Durable, scalable object storage |
| **Vector Database** | Qdrant file-based (single process) | Qdrant Cloud / self-hosted server | Supports concurrent workers & horizontal scaling |
| **Keyword Index** | `bm25_index.pkl` (local file) | Elasticsearch / OpenSearch | Distributed, sharded keyword search |
| **Parent Doc Store** | Pickle files (local disk) | Redis / DynamoDB | Low-latency key-value with replication |
| **Ingestion Pipeline** | `retriever_setup.py` (manual script) | Apache Airflow / AWS Lambda (event-driven) | Auto-triggers on new document upload |
| **Embedding Workers** | Local CPU (1 machine, ~1hr/1190 PDFs) | GPU instances (A10G via SageMaker) | 10-50x faster embedding generation |
| **API Server** | Single Uvicorn worker | Kubernetes (EKS/GKE) with HPA auto-scaling | Zero-downtime, handles traffic spikes |
| **Rate Limiting** | None | AWS API Gateway / Kong | Protects against API abuse |
| **Auth** | None | OAuth2 / JWT (API Gateway) | Secure multi-tenant access |
| **Monitoring** | Python `logging` | Prometheus + Grafana / Datadog | Latency, error rate, throughput dashboards |
| **CI/CD** | Manual `git push` | GitHub Actions → Docker → ECS/GKE | Automated test, build, deploy pipeline |

> **Code change required:** Only 2 lines in `core.py` — swap `QdrantClient(path=...)` for `QdrantClient(url=..., api_key=...)`. All retrieval, reranking, and generation logic stays identical.

---

### 📐 Scale & Cost Projections

Scaling calculations derived from live measured footprint: **579 MB Qdrant / 1,190 PDFs = 487 bytes/vector**.

| Scale Tier | Documents | Vectors | Qdrant Storage | Est. Monthly Infra Cost |
| :--- | ---: | ---: | ---: | ---: |
| **POC (Local)** | 1,190 | ~80K | 579 MB | **$0** |
| **Startup** | 10,000 | ~672K | ~4.8 GB | **~$80/mo** |
| **Small Firm** | 50,000 | ~3.4M | ~25 GB | **~$300/mo** |
| **Mid-Size Firm** | 500,000 | ~33.6M | ~245 GB | **~$1,800/mo** |
| **Enterprise** | 5,000,000 | ~336M | ~2.5 TB | **~$12,000/mo** |
| **Hyperscale** | 10,000,000 | ~672M | ~4.9 TB | **~$22,000/mo** |

---

### Per-Request API Cost Breakdown (At Scale)

**Current POC cost = $0.** All APIs used in this project have free tiers that cover development and demo usage:

| API | Free Tier Limit | Paid tier kicks in when... |
| :--- | :--- | :--- |
| **Groq** | 6,000 req/day, 500K tokens/day | > 6K queries/day or need SLA |
| **Cohere Rerank** | 1,000 calls/month | > 1K reranks/month |
| **HuggingFace Embeddings** | Free forever (runs locally) | Never — stays on CPU |

Once free limits are exceeded in production:

| API Call | Model | Cost Per 1M tokens | Avg tokens/request | Cost/request |
| :--- | :--- | :--- | :--- | :--- |
| **Query Compression** | Groq Llama 3.3 70B | ~$0.59 | ~150 tokens | ~$0.000089 |
| **LLM Generation** | Groq Llama 3.3 70B | ~$0.79 | ~2,000 tokens | ~$0.0016 |
| **Cohere Rerank** | Rerank v3.0 | $2.00 / 1K searches | 1 search | ~$0.002 |
| **Embeddings** | all-MiniLM-L6-v2 | $0 (local CPU) | — | **$0** |
| **Total per query** | | | | **~$0.004** |

> At 10,000 queries/day → **~$40/day in API costs**. Reduce by caching frequent queries (Redis) and batching Cohere rerank calls.

---

### 🔒 Enterprise Security Additions

| Layer | Implementation |
| :--- | :--- |
| **Transport** | TLS 1.3 (HTTPS enforced at API Gateway) |
| **Storage** | AES-256 at rest (S3 server-side encryption) |
| **Authentication** | OAuth2 + JWT (30-min token expiry) |
| **Document Access Control** | Qdrant payload filters for tenant isolation |
| **Audit Logging** | All queries logged to CloudWatch / BigQuery |
| **PII Scrubbing** | Pre-ingestion Lambda detects and redacts sensitive fields |
| **Rate Limiting** | 100 req/min per API key (API Gateway throttling) |

---

## 10 Million Document Scale Design

Numbers derived from measured POC footprint (579 MB Qdrant / 1,190 PDFs = 487 bytes/vector).

### Storage at 10M PDFs

| Component | Size | Service |
| :--- | :--- | :--- |
| Raw PDFs | ~13 TB | AWS S3 |
| Qdrant Vector DB | ~4.9 TB | Qdrant Cloud (10 shards, 2 replicas each) |
| BM25 Keyword Index | ~1.4 TB | Elasticsearch (3-node cluster) |
| Parent Doc Store | ~876 GB | DynamoDB |
| Vectors count | ~672M | 384-dim, all-MiniLM-L6-v2 |

### Ingestion at 10M PDFs

Single CPU script takes ~350 days. Distributed GPU pipeline:

| Step | Tool | Time |
| :--- | :--- | :--- |
| PDF upload trigger | S3 Event → SQS queue | instant |
| Text extraction | PyMuPDF workers (ECS pods) | parallel |
| Embedding generation | 100 × A10G GPU pods, batch=512 | ~84 hours |
| Vector upsert | Qdrant bulk API | included above |
| Parent chunk storage | DynamoDB bulk write | included above |

### Monthly Running Cost at 10M PDFs

| Component | Service | Cost/mo |
| :--- | :--- | :--- |
| PDF Storage | AWS S3 (13 TB) | ~$300 |
| Vector DB | Qdrant Cloud (4.9 TB) | ~$8,000 |
| Keyword Index | Elasticsearch (3 nodes) | ~$2,500 |
| Parent Doc Store | DynamoDB (876 GB) | ~$220 |
| API Server | Kubernetes EKS (5 pods) | ~$800 |
| LLM + Reranker APIs | Groq + Cohere (10K queries/day) | ~$1,200 |
| Cache | Redis ElastiCache | ~$120 |
| Monitoring | Datadog | ~$300 |
| **Total** | | **~$13,440/mo** |

### What changes in the code

Only 2 lines change in `core.py`. Everything else — chunking, retrieval logic, reranking, generation — stays identical.

```python
# POC (local file)
client = QdrantClient(path="./qdrant_db_local")

# 10M scale (server)
client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
```

BM25 moves from a `.pkl` file to Elasticsearch. Parent doc store moves from pickle files to DynamoDB. Query logic is untouched.

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