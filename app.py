import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 1. Download indexes from HF Datasets before loading the RAG core
import download_indexes
try:
    print("Downloading indexes from Hugging Face before starting server...")
    download_indexes.download_data()
except Exception as e:
    print(f"Failed to download indexes: {e}")

# 2. Now import the API app and Core (which will load the downloaded files)
from api import app

# 3. Mount a dummy Gradio interface just to satisfy Hugging Face's checks (Optional but safe)
import gradio as gr

def dummy_function():
    return "API is running!"

demo = gr.Interface(fn=dummy_function, inputs=None, outputs="text")
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    # Hugging Face Gradio spaces always look for port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
