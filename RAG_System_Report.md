# Advanced RAG System Project Report

## 1. Project Overview
This project successfully built an end-to-end Retrieval-Augmented Generation (RAG) system. The system can search through thousands of news articles instantly and use an advanced Large Language Model to answer questions based strictly on the retrieved context, complete with source citations.

## 2. The Dataset
* **Source:** We utilized the `CNN/DailyMail` dataset via the HuggingFace `datasets` library.
* **Volume:** We downloaded **5,000** full-length news articles.
* **Format:** Each article was saved as a distinct plain text file (`.txt`), resulting in 5,000 individual files.
* **Storage Location:** `./documents/`

## 3. The Embedding Model
* **Initial Attempt (Gemini API):** We initially attempted to use Google's Gemini API for embeddings. However, we discovered that Google's Free Tier has strict rate limits (100 requests per minute and 1,500 requests per day) where every single text chunk counts as a request. It was mathematically impossible to embed 20,000 chunks on the free tier.
* **Final Solution (Local AI):** We pivoted to the **HuggingFace `all-MiniLM-L6-v2`** model.
    * **Why we chose it:** It is a production-grade, highly efficient model that is incredibly small (only 22 MB). It runs 100% locally on the CPU.
    * **Result:** This allowed us to bypass all API rate limits and internet dependencies. We successfully processed and embedded all 5,000 documents in roughly **32 minutes** entirely on your local machine.

## 4. The Vector Database & Storage Architecture
We implemented the `ParentDocumentRetriever` architecture, which splits documents into smaller "child" chunks for highly accurate similarity searches, but returns the larger "parent" chunk to the AI to preserve context.

* **Vector Store (Qdrant):** The mathematical vector embeddings (384-dimensional) are stored in a local Qdrant database.
    * **Location:** `./qdrant_db_local/`
* **Document Store (PickleFileStore):** The actual text of the Parent Documents is serialized via Pickle and stored in a local file store.
    * **Location:** `./doc_store_local/`

## 5. The Application / Generation (`ap.py`)
* **Framework:** We used LangChain to orchestrate the retrieval and generation pipeline.
* **LLM:** We connected to the Google Gemini API using the `gemini-flash-latest` model to handle the final text generation.
* **Features:** 
    * Interactive terminal-based chat interface.
    * Built-in `try/except` safety catches to gracefully handle temporary `503` server overloads from Google.
    * Transparent source citations (`doc_xxxxx.txt`) appended to every answer so users can verify the AI's claims.
