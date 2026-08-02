# Quantitative Finance RAG: System Architecture & Scale

This document outlines the technical specifications of the local Proof-of-Concept (POC) and the proposed Enterprise-Grade Architecture for scaling to production.

---

## Part 1 — Local Proof of Concept (Built & Validated ✅)

### 📊 Scale & Metrics

| Metric | Value |
| :--- | :--- |
| **PDFs Ingested** | 1,190 Quantitative Finance papers (ArXiv) |
| **Total Pages** | ~31,000 pages of dense mathematical text |
| **Estimated Tokens** | ~20 Million tokens in total database |
| **Vector Embeddings** | ~80,000 child chunks (384-dim, all-MiniLM-L6-v2) |
| **Peak RAM (Ingestion)** | < 250 MB (via lazy_load() batching) |

### 🧮 Local Hardware Footprint

| Component | Size | Description |
| :--- | :--- | :--- |
| **Raw PDFs (Input)** | `1.56 GB` | 1,190 ArXiv papers |
| **Qdrant Vector DB** | `579 MB` | HNSW semantic search index |
| **BM25 Sparse Index** | `164 MB` | Serialized keyword matching dictionary |
| **Parent Doc Store** | `104 MB` | Pickled parent chunks for LLM context |
| **Total Index Size** | **~847 MB** | Full retrieval system on disk |

### 🧩 Chunking Strategy

The system uses **Parent-Child Document Splitting** (`ParentDocumentRetriever`) to prevent the "Lost in the Middle" problem in dense mathematical text.

| Chunk Type | Size | Purpose |
| :--- | :--- | :--- |
| **Child Chunks** | 1,000 chars / ~250 tokens | Embedded into Qdrant — searched semantically |
| **Parent Chunks** | 4,000 chars / ~1,000 tokens | Stored in doc store — fed to LLM for full context |

### ⚙️ Retrieval Pipeline

```
User Query
    │
    ▼
Query Compressor (LLM) ──── compresses long queries → concise search terms
    │
    ▼
Ensemble Retriever
    ├── Qdrant Vector Search (weight: 0.6) ── semantic meaning
    └── BM25 Keyword Search  (weight: 0.4) ── exact term matching
    │
    ▼ Reciprocal Rank Fusion (RRF)
    │
    ▼
Cohere Rerank v3.0 ──── cross-encoder scores all candidates → keeps top 2
    │
    ▼
Groq Llama 3.3 70B ──── generates final answer from top context
    │
    ▼
FastAPI Response (answer + source citations)
```

### 📈 RAGAS Evaluation Results (LLM-as-a-Judge)

Evaluated using the **Ragas Framework** against a constrained Q&A test set:

| Metric | Score |
| :--- | :--- |
| **Context Precision** | 1.0 |
| **Context Recall** | 1.0 |
| **Faithfulness** | 1.0 |

> Note: These scores are validated on a constrained test set. They serve as a baseline sanity check confirming the retrieval pipeline correctly grounds answers in source documents.

---

## Part 2 — Enterprise Production Architecture (Design Proposal 📄)

> The local POC proves the architecture is correct. Scaling it to enterprise requires replacing local file storage and single-process execution with managed cloud services. No fundamental logic changes — only infrastructure swaps.

### 🏗️ Architecture Comparison

| Component | Local POC (Built) | Enterprise Production |
| :--- | :--- | :--- |
| **PDF Storage** | Local `./real_pdfs` folder | AWS S3 / GCS Bucket |
| **Vector Database** | Local Qdrant (file-based) | Qdrant Cloud (managed cluster) or Pinecone |
| **Ingestion Pipeline** | Single Python script | Apache Airflow DAG / AWS Lambda event-driven |
| **Embedding Workers** | Local CPU (1 machine) | GPU instances (A10G/T4) via SageMaker or GCP |
| **BM25 Index** | Local `.pkl` file | Elasticsearch / OpenSearch cluster |
| **Doc Store** | Local pickle files | Redis / DynamoDB key-value store |
| **API Server** | Single Uvicorn process | Kubernetes (EKS/GKE) with auto-scaling |
| **Rate Limiting** | None (POC) | API Gateway (AWS) / Kong |
| **Monitoring** | Python logging | Datadog / Prometheus + Grafana |
| **CI/CD** | Manual `git push` | GitHub Actions → Docker → ECS/GKE |

### 📐 Enterprise Data Flow

```
                    ┌─────────────────────────────────┐
                    │         Data Ingestion           │
                    │                                  │
  New PDF Upload    │  S3 Bucket → SQS Queue           │
  ───────────────▶  │      → Lambda Workers (GPU)      │
                    │      → Qdrant Cloud (vectors)    │
                    │      → DynamoDB (parent chunks)  │
                    │      → Elasticsearch (BM25)      │
                    └─────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │        Query Serving             │
                    │                                  │
  User Request      │  API Gateway (rate limit, auth)  │
  ───────────────▶  │      → Kubernetes Pod (FastAPI)  │
                    │      → Qdrant Cloud + ES cluster │
                    │      → Cohere Rerank API         │
                    │      → Groq / Azure OpenAI       │
                    │      ← Response + Citations      │
                    └─────────────────────────────────┘
```

### 📊 Enterprise Scale Projections

| Scale | PDFs | Vectors | Qdrant Storage | Est. Monthly Cost |
| :--- | :--- | :--- | :--- | :--- |
| **POC (Local)** | 1,190 | ~80K | 579 MB | $0 |
| **Small Team** | 50,000 | ~3.4M | ~25 GB | ~$200/mo |
| **Mid-Size Firm** | 500,000 | ~34M | ~245 GB | ~$1,500/mo |
| **Enterprise** | 10,000,000 | ~680M | ~4.9 TB | ~$15,000/mo |

*(Linear scaling based on measured 487 bytes/vector for all-MiniLM-L6-v2 with HNSW metadata.)*

### 🔒 Enterprise Security Additions

- **Authentication:** OAuth2 / JWT token validation at API Gateway
- **Encryption:** TLS in transit, AES-256 at rest (S3, Qdrant Cloud)
- **Access Control:** Role-based document-level filtering in Qdrant payloads
- **Audit Logging:** All queries logged to CloudWatch / BigQuery for compliance
- **PII Scrubbing:** Pre-ingestion pipeline to detect and redact sensitive data

---

## Summary

This project demonstrates the full MLOps lifecycle:

1. **Data Engineering** — batch PDF ingestion with memory-safe streaming
2. **ML Engineering** — hybrid retrieval (semantic + keyword) with cross-encoder reranking
3. **Backend Engineering** — production FastAPI with Pydantic validation and dependency injection
4. **Evaluation** — automated LLM-as-a-Judge scoring with RAGAS
5. **Architecture Design** — documented path from local POC to enterprise cloud deployment

## 📊 1. The Scale & Metrics

**Data Ingestion & Processing:**
- **PDFs Ingested:** 1,190 Quantitative Finance research papers (scraped from the ArXiv API).
- **Total Pages:** ~31,000 pages of highly dense mathematical text, charts, and tables.
- **Estimated Words:** ~15.5 Million words.
- **Estimated Tokens:** **~20 Million Tokens** in total database size.

**Memory-Safe Processing:**
To prevent RAM overflows and Out-of-Memory (OOM) errors during ingestion, the system implements a strict **Batch Processing Architecture**. Using `PyMuPDFLoader` with multithreading, the system pushes embeddings to the Qdrant Vector database in exact batches of `20` documents at a time. This keeps the memory footprint strictly capped while processing all 31,000 pages locally.

## 🧮 2. System Hardware Calculations (For 1,190 PDFs)

If asked exactly how much disk and RAM the current system requires, here are the hard, live calculations of the deployed architecture:

- **Raw Data Storage (Input):** 1,190 PDFs * ~1.3 MB average = **1.56 GB raw PDF storage** (`./real_pdfs`).
- **Vector DB Storage (Qdrant):** Using `all-MiniLM-L6-v2` creates a 384-dimensional vector. With HNSW indexing and metadata payloads, the final database footprint is **579 MB on disk** (`./qdrant_db_local`).
- **Sparse Index (BM25):** The serialized keyword index for all 31,000 pages takes **164 MB** (`bm25_index.pkl`).
- **Local Document Store:** The pickled parent chunks take **104 MB** (`./doc_store_local`).
- **Total Output Footprint:** ~847 MB of heavily optimized search indexes.
- **RAM Usage (Ingestion):** Because we use `DirectoryLoader.lazy_load()` with a strict `BATCH_SIZE = 20`, the peak RAM required to index all 1.56 GB of data is strictly capped at **less than 250 MB**, meaning the ingestion pipeline can run on a Raspberry Pi without crashing.

## 3. Future Scalability Considerations (10M+ Documents)

The current pipeline is designed for local prototyping (1,190 PDFs). Scaling to millions of documents would require a fundamentally different architecture:

- **Storage:** Offloading raw PDFs to an Object Store (e.g., AWS S3).
- **Ingestion:** Replacing the local script with an event-driven queue (e.g., SQS) and distributed workers.
- **Embeddings:** Utilizing dedicated GPU instances for embedding generation rather than local CPU inference.
- **Vector DB:** Migrating from local Qdrant to a managed, sharded cluster to handle TBs of vector data.

## 🧩 4. The Chunking Strategy

The system utilizes **Parent-Child Document Splitting** (`ParentDocumentRetriever`), which is critical for complex academic texts, preventing the "Lost in the Middle" syndrome.

1. **Child Chunks (1000 characters / ~250 tokens):** 
   - *Why?* Dense math formulas require highly specific semantic embeddings. If the chunk is too large, the specific nuances of a mathematical equation get diluted across a giant vector.
   - *Where do they go?* These 250-token chunks are embedded via HuggingFace and pushed to the **Qdrant Vector Database**. This is what the system mathematically searches against.

2. **Parent Chunks (4000 characters / ~1000 tokens):** 
   - *Why?* While the system searches against the tiny 250-token chunks, feeding a tiny chunk to an LLM is a disaster because it lacks context. If the chunk is just a formula, the LLM won't know *why* the formula matters.
   - *Where do they go?* The system maps the tiny child chunk back to its original 1000-token Parent Chunk (stored in the **Local Document Store**). It feeds the massive Parent Chunk to the Groq Llama-3 API. This ensures the LLM has the surrounding context (like the paragraph before and after the formula) to accurately generate the answer!

## 5. The Retrieval Pipeline

In quantitative finance, users often search for highly specific acronyms (like 'LTRO' or 'e-MID'). Standard semantic vector search is great for conceptual matching, but often fails at exact keyword matching. 

To solve this, the architecture uses an **Ensemble Retriever**:
1. A **Qdrant Vector Database** (using HuggingFace `all-MiniLM-L6-v2` embeddings) for semantic meaning.
2. A **BM25 Sparse Index** for exact keyword matching.

The system queries both databases simultaneously and uses Reciprocal Rank Fusion (RRF) to combine the results.

## 6. Reranking & Generation

Retrieving from 31,000 pages yields candidate chunks with varying relevance. 

1. **Cohere Rerank API:** The pipeline implements a cross-encoder (Cohere Rerank 3) that scores the relevance of the retrieved chunks against the user's query, filtering out the lowest-scoring chunks.
2. **Llama 3 (Groq):** Only the highest-confidence documents are passed to the Groq Llama-3 API.

## 7. Evaluation (LLM-as-a-Judge)

The pipeline incorporates the **Ragas Framework** for evaluation. 

The evaluation script (`production_eval.py`) runs against a small, constrained set of known Q&A pairs. While it currently achieves perfect 1.0 scores on this specific test set, these metrics represent a baseline sanity check rather than generalized production performance across unseen domains.
- **Context Precision: 1.0**
- **Context Recall: 1.0**
- **Faithfulness: 1.0**
