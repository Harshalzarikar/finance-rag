# 🚀 Autonomous Agentic RAG (Finance Domain)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-orange.svg)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Database-purple.svg)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3-green.svg)

An enterprise-grade, **Self-Querying Retrieval-Augmented Generation (RAG)** system designed to scale across thousands of financial documents without context dilution. 

This project solves the "Needle in a Haystack" problem by implementing a dynamic **LLM Query Router** that intercepts user queries, deduces the intended target entity (e.g., a specific company), and autonomously applies fuzzy metadata filters to a Hybrid Search database.

---

## ✨ Key Features

* **🧠 Autonomous LLM Query Router:** Uses Llama 3 via Groq to instantly analyze a user's prompt and extract specific filtering criteria before vector search occurs.
* **🔍 Hybrid Search Architecture:** Combines semantic vector search (**Qdrant**) with keyword search (**BM25**) via an `EnsembleRetriever`, fused together using Reciprocal Rank Fusion (RRF).
* **🏷️ Dynamic Regex Ingestion:** Automatically parses massive text corpuses during ingestion, extracting company names and injecting them as searchable payload metadata.
* **🎯 Fuzzy Metadata Matching:** Uses Qdrant's `MatchText` to gracefully handle naming inconsistencies (e.g., matching "Apple" to "Apple Inc.").
* **⚖️ LLM-as-a-Judge Evaluation Suite:** Includes an automated testing pipeline to strictly grade the system on Faithfulness, Relevance, Precision, and Recall.

---

## 🏗️ System Architecture

### 1. Ingestion Pipeline (`retriever_setup.py`)
1. **Directory Loader:** Ingests 5,000+ financial text documents.
2. **Regex Processor:** Scans the first line of every document to extract the Company Name.
3. **Parent-Child Chunking:** Splits documents into 4000-character parent chunks and 1000-character child chunks.
4. **Vectorization:** Embeds child chunks using local `all-MiniLM-L6-v2` embeddings and stores them in Qdrant with the extracted Company metadata.

### 2. Retrieval & Generation (`ap.py`)
1. **Query Interception:** User asks a question (e.g., *"What is Apple's revenue?"*).
2. **LLM Routing:** A rapid Groq LLM call deduces the company filter (`Apple`) or falls back to `NONE`.
3. **Hybrid Search:** Qdrant instantly slices the 5,000-document database using the `Apple` filter. BM25 simultaneously performs a keyword search.
4. **RRF & Reranking:** Results are fused, and a final LLM reranking step drops irrelevant context.
5. **Generation:** Llama 3 generates a highly accurate, hallucination-free response based strictly on the isolated context.

---

## 📊 Performance Metrics

Evaluated against a 5,000-document synthetic financial dataset (~670,000 tokens) using the automated `evaluate.py` test suite.

| Metric | Score | Analysis |
|---|:---:|---|
| **Answer Relevance** | **10.0 / 10** | When data is successfully retrieved, the generation perfectly answers the query. |
| **Context Recall** | **9.7 / 10** | The retriever successfully captures nearly all factual ground-truth points required. |
| **Faithfulness** | **9.3 / 10** | The Generator strictly adheres to the retrieved text and refuses to hallucinate. |
| **Context Precision** | **8.3 / 10** | The Qdrant Metadata Filtering successfully isolates exact documents out of thousands. |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Groq API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Harshalzarikar/finance-rag.git
   cd finance-rag
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

### Usage

**1. Build the Database (Ingestion):**
```bash
python retriever_setup.py
python build_bm25.py
```

**2. Chat with the Agent:**
```bash
python ap.py
```

**3. Run the Evaluation Suite:**
```bash
python evaluate.py
```

---

## 📁 Repository Structure

```text
finance-rag/
├── ap.py                      # Main application, Query Router, and Generation loop
├── retriever_setup.py         # Vector DB ingestion and dynamic metadata extraction
├── build_bm25.py              # Keyword index builder
├── evaluate.py                # LLM-as-a-Judge automated testing pipeline
├── generate_finance_docs.py   # Script to generate synthetic testing data
├── RAG_Evaluation_Report.md   # Deep-dive into the evaluation methodology and results
└── requirements.txt           # Project dependencies
```
