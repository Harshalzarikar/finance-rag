import os
import sys
import types
import json
import pandas as pd
from datasets import Dataset

# Patch Ragas Bug with langchain_community 0.3.0
if "langchain_community.chat_models.vertexai" not in sys.modules:
    dummy_module = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_module.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_module
    
from dotenv import load_dotenv

# Ragas imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)

# Import our custom RAG architecture
from ap import setup_rag, cohere_rerank, dynamic_retrieve
from langchain_core.prompts import PromptTemplate

load_dotenv()

def main():
    print("==================================================")
    print("PRODUCTION RAG EVALUATION SUITE (Ragas)")
    print("==================================================")
    
    # 1. Initialize RAG components
    components, llm = setup_rag()
    
    # 2. Test Questions with Ground Truth Answers for Quantitative Finance
    test_cases = [
        {
            "question": "What are the most common models used for predicting Bid-Ask spread conditional distributions?",
            "ground_truth": "The methodology introduces a Hierarchical Correlation Reconstruction (HCR) model for predicting conditional probability distributions of bid-ask spreads, while mentioning simpler predictors like AMI and HLR."
        },
        {
            "question": "How does the Kyle Single Period model handle insider trading?",
            "ground_truth": "In Kyle's single period model, an insider attempts to maximize their profit by strategically trading on private information while a market maker sets prices to ensure zero expected profit."
        },
        {
            "question": "What is the Heston model used for?",
            "ground_truth": "The Heston model is a mathematical model used in quantitative finance to price options, specifically by assuming that volatility is stochastic rather than constant."
        }
    ]
    
    # 3. The Generation Prompt
    generation_prompt = PromptTemplate.from_template(
        "You are an AI assistant answering questions based on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer the question using ONLY the provided context. If the answer is not in the context, say 'I don't know'."
    )
    
    print("\nRunning RAG Pipeline to generate answers and retrieve contexts...\n")
    
    data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": []
    }
    
    for i, test in enumerate(test_cases):
        q = test["question"]
        gt = test["ground_truth"]
        print(f"[{i+1}/{len(test_cases)}] Generating answer for: '{q}'")
        
        # --- Run Retrieval DYNAMICALLY ---
        docs = dynamic_retrieve(q, components, llm)
        
        # Run Cohere Reranking (Filtering)
        docs = cohere_rerank(docs, q)
        
        contexts = []
        if not docs:
            answer = "I don't know."
        else:
            contexts = [doc.page_content for doc in docs]
            context_text = "\n---\n".join(contexts)
            formatted_gen = generation_prompt.format(context=context_text, question=q)
            content = llm.invoke(formatted_gen).content
            if isinstance(content, list):
                answer = " ".join([b.get("text", "") for b in content if isinstance(b, dict)]).strip()
            else:
                answer = content.strip()
            
        data["user_input"].append(q)
        data["response"].append(answer)
        data["retrieved_contexts"].append(contexts)
        data["reference"].append(gt)
    
    # 4. Build HuggingFace Dataset
    try:
        dataset = Dataset.from_dict(data)
    except Exception as e:
        print(f"Failed to build dataset: {e}")
        return

    print("\nInitializing Ragas Evaluation Metrics...")
    
    # Configure Ragas to use our specific LLM and Embeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    eval_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    metrics = [
        faithfulness,
        context_precision,
        context_recall
    ]

    print("Evaluating Architecture (This may take a minute)...\n")
    try:
        from ragas.run_config import RunConfig
        
        # We removed max_workers=1 so Groq can process all 12 metrics at lightning speed!
        result = evaluate(
            dataset = dataset,
            metrics = metrics,
            llm = llm,
            embeddings = eval_embeddings,
            raise_exceptions=False
        )
        
        df = result.to_pandas()
        
        print("==================================================")
        print("FINAL RAGAS SCORECARD (Pandas DataFrame)")
        print("==================================================")
        print(df.to_string())
        
        # Save to CSV for CI/CD pipelines
        df.to_csv("ragas_evaluation_results.csv", index=False)
        print("\nResults successfully saved to 'ragas_evaluation_results.csv'!")
        
    except Exception as e:
        print(f"\n[Error] Ragas Evaluation Failed: {e}")
        print("Note: Ragas can sometimes be strict with non-OpenAI API schemas. Ensure Groq/Gemini models are supported in your version.")

if __name__ == "__main__":
    main()
