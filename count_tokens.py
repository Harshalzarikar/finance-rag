import os
import glob
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
except ImportError:
    enc = None

def main():
    docs = glob.glob("documents/*.txt")
    total_chars = 0
    total_words = 0
    total_tokens = 0
    max_tokens = 0
    min_tokens = float('inf')
    
    for doc_path in docs:
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
            chars = len(text)
            words = len(text.split())
            if enc:
                tokens = len(enc.encode(text))
            else:
                tokens = words * 1.3 # rough estimation
            
            total_chars += chars
            total_words += words
            total_tokens += tokens
            max_tokens = max(max_tokens, tokens)
            min_tokens = min(min_tokens, tokens)
            
    print(f"Total Documents: {len(docs)}")
    print(f"Total Chars: {total_chars}")
    print(f"Total Words: {total_words}")
    print(f"Total Tokens: {int(total_tokens)}")
    print(f"Avg Tokens/Doc: {int(total_tokens / len(docs))}")
    print(f"Max Tokens/Doc: {int(max_tokens)}")
    print(f"Min Tokens/Doc: {int(min_tokens)}")

if __name__ == "__main__":
    main()
