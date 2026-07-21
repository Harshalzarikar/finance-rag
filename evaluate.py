import os
import json
from langchain_core.prompts import PromptTemplate
from ap import setup_rag, llm_rerank, dynamic_retrieve

def main():
    print("==================================================")
    print("RAG System Evaluation Tool (LLM-as-a-Judge)")
    print("==================================================")
    
    # 1. Initialize RAG components
    components, llm = setup_rag()
    
    # 2. Test Questions with Ground Truth Answers
    test_cases = [
        {
            "question": "Have any companies announced mergers or acquisitions recently?",
            "ground_truth": "Various companies have entered into definitive agreements to acquire other firms in all-cash transactions to expand market share."
        },
        {
            "question": "What is the latest news on Apple's Q3 2023 financials?",
            "ground_truth": "Apple announced Q3 2023 revenue of $81.8 billion, driven by an all-time revenue record in Services and strength in emerging markets."
        },
        {
            "question": "What happened to the escaped prisoner?",
            "ground_truth": "The escaped prisoner, Feigley, escaped from a Pennsylvania prison in 1976 and was recaptured."
        }
    ]
    
    # 3. The Generation Prompt
    generation_prompt = PromptTemplate.from_template(
        "You are an AI assistant answering questions based on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer the question using ONLY the provided context. If the answer is not in the context, say 'I don't know'."
    )
    
    # 4. The Evaluation Prompt (Grading all 4 RAG metrics)
    eval_prompt = PromptTemplate.from_template(
        "You are an expert AI evaluator grading a RAG system.\n"
        "Score the system based on the Question, Context, Generated Answer, and Ground Truth Answer.\n\n"
        "Question: {question}\n"
        "Ground Truth: {ground_truth}\n"
        "Retrieved Context: {context}\n"
        "Generated Answer: {answer}\n\n"
        "Evaluate based on four metrics (0 to 10):\n"
        "1. Faithfulness: Is the generated answer derived ONLY from the context? (0 = hallucinated, 10 = perfectly backed by context)\n"
        "2. Relevance: Does the generated answer directly address the question? (0 = irrelevant, 10 = perfect)\n"
        "3. Context Precision: Are the retrieved context documents highly relevant to the question? (0 = totally useless context, 10 = context is exactly what is needed)\n"
        "4. Context Recall: Did the retrieved context contain ALL the facts present in the Ground Truth? (0 = missing all facts, 10 = context contains all ground truth facts)\n\n"
        "Return EXACTLY this JSON format:\n"
        '{{"faithfulness": 8, "relevance": 9, "context_precision": 7, "context_recall": 10, "reasoning": "Brief explanation..."}}'
    )

    total_faith = 0
    total_rel = 0
    total_prec = 0
    total_rec = 0
    successful_tests = 0

    print("\nRunning Evaluation Suite...\n")
    
    for i, test in enumerate(test_cases):
        q = test["question"]
        gt = test["ground_truth"]
        print(f"Test {i+1}: {q}")
        
        # --- Run Retrieval DYNAMICALLY ---
        docs = dynamic_retrieve(q, components, llm)
        
        # Run LLM Reranking (Filtering)
        docs = llm_rerank(docs, q, llm)
        
        if not docs:
            print("  [X] No relevant documents found.")
            # If no docs are found, recall/precision are 0.
            context_text = "NONE"
            answer = "I don't know."
        else:
            context_text = "\n---\n".join([doc.page_content for doc in docs])
            formatted_gen = generation_prompt.format(context=context_text, question=q)
            answer = llm.invoke(formatted_gen).content.strip()
        
        # Evaluate
        formatted_eval = eval_prompt.format(question=q, context=context_text, answer=answer, ground_truth=gt)
        eval_result = llm.invoke(formatted_eval).content.strip()
        
        try:
            if "```json" in eval_result:
                eval_result = eval_result.split("```json")[1].split("```")[0].strip()
            elif "```" in eval_result:
                eval_result = eval_result.split("```")[1].strip()
                
            scores = json.loads(eval_result)
            faith = scores.get("faithfulness", 0)
            rel = scores.get("relevance", 0)
            prec = scores.get("context_precision", 0)
            rec = scores.get("context_recall", 0)
            
            print(f"  > Generated Answer Preview: {answer[:75]}...")
            print(f"  > [Generator] Faithfulness: {faith}/10 | Relevance: {rel}/10")
            print(f"  > [Retriever] Precision:    {prec}/10 | Recall:    {rec}/10")
            print(f"  > Evaluator Reason: {scores.get('reasoning')}\n")
            
            total_faith += faith
            total_rel += rel
            total_prec += prec
            total_rec += rec
            successful_tests += 1
        except Exception as e:
            print(f"  > Failed to parse evaluation: {e}")
            print(f"  > Raw Eval: {eval_result}\n")

    print("==================================================")
    print("FINAL RAG SCORECARD")
    print("==================================================")
    if successful_tests > 0:
        print(f"Average Faithfulness:      {total_faith / successful_tests:.1f} / 10")
        print(f"Average Answer Relevance:  {total_rel / successful_tests:.1f} / 10")
        print(f"Average Context Precision: {total_prec / successful_tests:.1f} / 10")
        print(f"Average Context Recall:    {total_rec / successful_tests:.1f} / 10")
    else:
        print("No tests completed successfully.")
    print("==================================================")

if __name__ == "__main__":
    main()
