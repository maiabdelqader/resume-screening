"""
Main entry point for executing the Retrieval-Augmented Generation (RAG) pipeline.

This script performs batch resume screening by:
- Loading a LLaMA-based or OpenAI-based model for both subquery and final response generation.
- Employing a hybrid retrieval strategy by combining semantic search (FAISS) with keyword matching (TF-IDF).
- Running in basic or advanced mode with score-aware fusion of sub-query results.
- Selecting the most relevant evidence from retrieved resumes to improve grounding.
- Saving results for multiple test sets and evaluating them with RAGAS metrics.
"""

import logging
import os

from dotenv import load_dotenv

from rag_pipeline    import RAGPipeline
from ragas_evaluator import RAGASEvaluator
from resume_engine   import ResumeEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables from a .env file
load_dotenv("../.env")

# Load key configuration paths, model identifiers, and API tokens from environment variables
faiss_path, resumes_path, gpt3_turbo_model, llama2_13b_model, llama3_8b_model, embedding_model, huggingface_token = map(
    os.getenv,
    ("FAISS_PATH", "RESUMES_PATH", "GPT3_TURBO_MODEL", "LLAMA2_13B_MODEL", "LLAMA3_8B_MODEL", "EMBEDDING_MODEL", "HUGGINGFACE_TOKEN")
)

# RAG Mode: "basic" or "advanced"
rag_mode = "advanced"

# Define the output directory based on the selected RAG mode
output_dir = f"../data/{rag_mode}-rag"
os.makedirs(output_dir, exist_ok=True)

def main() -> None:
    """
    Run the RAG pipeline and evaluate its performance across all test sets.
    """
    logging.info(f"🚀 Mode: {rag_mode.upper()} | Output: {output_dir}")

    # Initialize the resume retrieval engine, which loads or builds the FAISS index
    resume_engine = ResumeEngine(
        model_name=embedding_model,
        faiss_path=faiss_path,
        resumes_path=resumes_path
    )

    # Initialize the main RAG pipeline with the resume engine and specified model
    rag_pipeline = RAGPipeline(
        resume_engine=resume_engine,
        huggingface_token=huggingface_token,
        rag_mode=rag_mode,
        model_name=llama3_8b_model
    )

    # Initialize the RAGAS evaluator for assessing the pipeline's performance
    ragas_evaluator = RAGASEvaluator()

    # Iterate over test-set indices (1..5), constructing input/output paths on the fly
    for test_path, output_path in (
        (f"../data/test-sets/testset-{i}.csv", f"{output_dir}/result-{i}.csv")
        for i in range(1, 6)
    ):
        # Run the RAG pipeline for the current test set
        logging.info(f"🧪 Running RAG pipeline on: {test_path}")
        rag_pipeline.run(test_csv_path=test_path, output_csv_path=output_path, top_k=3)

        # Construct the path for the evaluation results file
        file_name = os.path.basename(output_path)
        eval_path = os.path.join(output_dir, f"evaluation-{file_name}")

        # Run the RAGAS evaluation on the generated results
        logging.info(f"🧪 Running RAGAS evaluation for: {output_path}")
        ragas_evaluator.evaluate_from_file(result_path=output_path, output_eval_path=eval_path)

if __name__ == "__main__":
    main()
