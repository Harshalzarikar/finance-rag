from ap import PickleFileStore, STORE_DIR

def main():
    store = PickleFileStore(STORE_DIR)
    apple_docs = 0
    escaped_docs = 0
    
    for key in store.yield_keys():
        docs = store.mget([key])
        if docs and docs[0]:
            doc = docs[0]
            if "Apple" in doc.page_content:
                apple_docs += 1
                print("--- Found Apple Document ---")
                print(f"Content: {doc.page_content[:100]}")
            if "Feigley" in doc.page_content:
                escaped_docs += 1
                
    print(f"Apple Parent Docs: {apple_docs}")
    print(f"Escaped Prisoner Parent Docs: {escaped_docs}")

if __name__ == "__main__":
    main()
