import arxiv
import os
import time
import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
@retry(
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def download_pdf(url: str, filepath: str):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(filepath, 'wb') as f:
        f.write(response.content)

def main():
    # Number of PDFs to download (Configurable)
    MAX_RESULTS = 1200
    SAVE_DIR = "./real_pdfs"
    
    # Create the directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"Searching ArXiv for top {MAX_RESULTS} Quantitative Finance (q-fin) papers...")
    
    # Construct the default API client
    client = arxiv.Client()
    
    # Search for Quantitative Finance papers
    search = arxiv.Search(
        query="cat:q-fin.*",
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = list(client.results(search))
    print(f"Found {len(papers)} papers! Starting download...\n")
    
    for i, paper in enumerate(papers):
        # Create a safe filename using the ArXiv ID
        filename = f"{paper.get_short_id()}.pdf"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"[{i+1}/{len(papers)}] {filename} already exists. Skipping.")
            continue
            
        print(f"[{i+1}/{len(papers)}] Downloading: {paper.title[:60]}...")
        
        try:
            download_pdf(paper.pdf_url, filepath)
            # Sleep for 1 second to respect ArXiv API rate limits
            time.sleep(1.0)
        except Exception as e:
            print(f"  [Error] Failed to download {filename} after retries: {e}")
            
    print(f"\nSuccessfully downloaded papers to {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    main()
