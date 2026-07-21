from ap import PickleFileStore, STORE_DIR

def main():
    store = PickleFileStore(STORE_DIR)
    
    for key in store.yield_keys():
        docs = store.mget([key])
        if docs and docs[0]:
            doc = docs[0]
            if "Apple" in doc.page_content:
                print(f"APPLE METADATA: {doc.metadata}")
            if "Feigley" in doc.page_content:
                print(f"PRISONER METADATA: {doc.metadata}")

if __name__ == "__main__":
    main()
