import pickle

def main():
    print("Loading BM25 Keyword Index...")
    try:
        with open("bm25_index.pkl", "rb") as f:
            bm25_retriever = pickle.load(f)
        print("✅ BM25 Index Loaded Successfully!")
    except FileNotFoundError:
        print("❌ Error: bm25_index.pkl not found. Please run build_bm25.py first.")
        return
    
    print("\n" + "="*50)
    print("BM25 (Keyword-Only) Testing Tool")
    print("This will bypass the semantic AI and ONLY use exact keyword matching.")
    print("="*50)

    while True:
        try:
            query = input("\nEnter a keyword to search (or 'quit' to exit): ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        docs = bm25_retriever.invoke(query)
        
        if not docs:
            print("No documents found containing those exact keywords.")
            continue
            
        print(f"\nTop {len(docs)} documents found purely by EXACT KEYWORD MATCH:")
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            print(f"\n[{i+1}] Source: {source}")
            print(f"Snippet: {doc.page_content[:200]}...")
            print("-" * 50)

if __name__ == "__main__":
    main()
