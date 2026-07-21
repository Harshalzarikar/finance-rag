from qdrant_client import QdrantClient
from ap import QDRANT_DB_DIR

def main():
    client = QdrantClient(path=QDRANT_DB_DIR)
    
    # Let's count how many documents have company="Apple"
    from qdrant_client.http import models as rest
    
    count_res = client.count(
        collection_name="parent_document_store",
        count_filter=rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.company",
                    match=rest.MatchValue(value="Apple")
                )
            ]
        )
    )
    print(f"Docs with company='Apple': {count_res.count}")
    
    # Let's fetch some random unique companies to see what they look like
    res = client.scroll(
        collection_name="parent_document_store",
        limit=100
    )
    companies = set([p.payload.get("metadata", {}).get("company") for p in res[0]])
    print(f"Sample companies found: {companies}")

if __name__ == "__main__":
    main()
