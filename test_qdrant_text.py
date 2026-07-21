from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from ap import QDRANT_DB_DIR

def main():
    client = QdrantClient(path=QDRANT_DB_DIR)
    
    # Try MatchText
    try:
        res = client.scroll(
            collection_name="parent_document_store",
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.company",
                        match=rest.MatchText(text="Apple")
                    )
                ]
            )
        )
        print("MatchText Results:", len(res[0]))
    except Exception as e:
        print("MatchText Failed:", e)
        
if __name__ == "__main__":
    main()
