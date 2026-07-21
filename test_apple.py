import os
from ap import setup_rag, dynamic_retrieve

def main():
    components, llm = setup_rag()
    query = "What is the latest news on Apple's Q3 2023 financials?"
    docs = dynamic_retrieve(query, components, llm)
    
    print(f"\nRetrieved {len(docs)} docs:")
    for i, doc in enumerate(docs):
        print(f"--- Doc {i} ---")
        print(f"Metadata: {doc.metadata}")
        print(doc.page_content[:200])
        print("...")

if __name__ == "__main__":
    main()
