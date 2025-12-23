"""
rag_pipeline.py

Implements a flexible Retrieval-Augmented Generation (RAG) pipeline for resume screening, supporting both basic and advanced retrieval strategies.

Key Features:
- Retrieves top-K resumes relevant to a job description using FAISS embeddings, with optional hybrid retrieval.
- Supports both basic (single-query) and advanced (multi-subquery with score-aware fusion) modes.
- Uses either LLaMA-based or OpenAI-based models for generating structured recruiter-style responses.
- Includes subquery caching and optimization to improve retrieval diversity and reduce redundancy.
- Applies score-aware Reciprocal Rank Fusion (RRF) to combine results from multiple subqueries.
- Saves results (job descriptions, resumes, and generated responses) to CSV for evaluation and analysis.
"""

import gc
import os
import re
import math
import torch
import logging
import hashlib

import numpy  as np
import pandas as pd

from collections import defaultdict
from typing import List, Dict

from llama_pipeline     import LlamaPipeline
from subquery_optimizer import SubqueryOptimizer

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


class RAGPipeline:
    def __init__(
        self,
        resume_engine,
        huggingface_token: str,
        rag_mode: str,
        model_name: str
    ) -> None:
        """
        Initialize the RAG pipeline components.

        Args:
            resume_engine: Resume retrieval engine (FAISS-based) used to fetch relevant resumes.
            huggingface_token (str): Authentication token for accessing Hugging Face models.
            rag_mode (str): Retrieval mode, either "basic" or "advanced".
            model_name (str): Identifier of the language model to use for response generation (LLaMA or GPT-based).
        """
        self.resume_engine = resume_engine
        self.rag_mode = rag_mode

        # Load the summarization LLM model
        if "gpt" in model_name.lower():
            self.summary_model = OpenAIPipeline(model_name=model_name, api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.summary_model = LlamaPipeline(model_name=model_name, huggingface_token=huggingface_token)
            self.summary_model.load()

        # Load the subquery decomposition model if using advanced mode
        self.subquery_model = None
        self.subquery_cache = pd.DataFrame()
        self.subquery_cache_path = None

        if self.rag_mode == "advanced":
            self.subquery_model = self.summary_model

            # Sanitize model name to create a valid filename for the cache
            sanitized_model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
            self.subquery_cache_path = f"subquery_cache_{sanitized_model_name}.csv"

            # Load existing cache if it exists
            if os.path.exists(self.subquery_cache_path):
                self.subquery_cache = pd.read_csv(self.subquery_cache_path)
            else:
                self.subquery_cache = pd.DataFrame(columns=['job_hash', 'subqueries'])                

        # Load the same embedding model used in FAISS engine
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.resume_engine.model_name,
            model_kwargs={"device": torch.device("cuda")}
        )

        # Initialize subquery optimizer to rank subqueries
        self.subquery_optimizer = SubqueryOptimizer(         
            self.embedder, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0
        )

    def _get_or_generate_subqueries(self, job_description: str) -> List[str]:
        """
        Retrieves subqueries from cache or generates them if not found.

        Args:
            job_description (str): The job description.

        Returns:
            List[str]: A list of generated or cached subquery strings. Returns an empty list if generation fails.
        """
        # Create a unique hash for the job description to use as a cache key
        job_hash = hashlib.md5(job_description.encode()).hexdigest()
        
        # Ensure the 'job_hash' column exists for querying
        if 'job_hash' not in self.subquery_cache.columns:
            self.subquery_cache['job_hash'] = None

        # Check if an entry with this hash already exists in the cache
        cached_entry = self.subquery_cache[self.subquery_cache['job_hash'] == job_hash]
        
        # If a cached entry is found, load and return it.
        if not cached_entry.empty:
            logging.info(f"CACHE HIT for job description hash: {job_hash}")
            subqueries_str = cached_entry.iloc[0]['subqueries']

            # Split the cached string by the delimiter to reconstruct the list
            return subqueries_str.split('|||') if pd.notna(subqueries_str) else []
        
        # If no cache entry is found, generate new subqueries.
        logging.info(f"CACHE MISS for job description hash: {job_hash}. Generating new subqueries...")

        subqueries = self.subquery_model.generate_subqueries(job_description)

        # If subqueries were successfully generated, save them to the cache
        if subqueries:
            self._cache_subqueries(job_hash, subqueries)

        return subqueries

    def _cache_subqueries(self, job_hash: str, subqueries: List[str]):
        """
        Saves a new set of generated subqueries to the cache.

        Args:
            job_hash (str): The MD5 hash of the job description.
            subqueries (List[str]): The list of subquery strings to cache.
        """
        # Join the list of subqueries into a single string using a unique delimiter
        subqueries_str = '|||'.join(subqueries)

        # Create a new DataFrame row with the hash and the subquery string
        new_entry = pd.DataFrame([{'job_hash': job_hash, 'subqueries': subqueries_str}])

        # Append the new entry to the in-memory cache.
        self.subquery_cache = pd.concat([self.subquery_cache, new_entry], ignore_index=True)

        # Write the updated cache back to the CSV file for persistence
        self.subquery_cache.to_csv(self.subquery_cache_path, index=False)

    @staticmethod
    def _softmax(values, temperature: float = 0.8) -> List[float]:
        """
        Compute softmax-normalized weights from a list of values.
    
        Args:
            values (list[float]): List of numeric values to normalize.
            temperature (float): Softmax temperature (default: 0.8).
    
        Returns:
            list[float]: Normalized weights corresponding to each input value.
        """
        if not values:
            return []
    
        # Shift values by the maximum value for numerical stability
        max_value = max(values)
    
        # Apply temperature scaling and exponentiation
        exp_values = [
            math.exp((val - max_value) / max(1e-6, temperature))
            for val in values
        ]
    
        # Normalize so results sum to 1
        sum_exp = sum(exp_values) or 1.0
        normalized_weights = [exp_val / sum_exp for exp_val in exp_values]
    
        return normalized_weights

    def score_aware_fusion(
        self,
        subquery_info: List[Dict],
        k: int = 40,
        rank_weight: float = 0.35
    ) -> Dict[str, float]:
        """
        Applies score-aware Reciprocal Rank Fusion (RRF) to combine ranked results from multiple subqueries.
    
        Args:
            subquery_info (List[Dict]): A list of dictionaries, each with a 'weight' and 'resume_scores' key.
            k (int): RRF smoothing constant.
            rank_weight (float): Weight controlling the blend between rank and score.
    
        Returns:
            Dict[str, float]: Fused resume_id → final score (sorted descending).
        """
        if not subquery_info:
            return {}

        # Compute normalized weights for each subquery based on optimizer scores
        subquery_weights   = [subquery_result.get("weight", 0) for subquery_result in subquery_info]
        normalized_weights = self._softmax(subquery_weights, temperature=0.8)
        
        fused_scores = defaultdict(float)
    
        for weight, subquery_result in zip(normalized_weights, subquery_info):
            resume_scores = subquery_result.get("resume_scores", {})
            
            # Sort resumes by their similarity score (highest first)
            ranked_resumes = sorted(resume_scores.items(), key=lambda x: x[1], reverse=True)

            for rank, (resume_id, similarity_score) in enumerate(ranked_resumes, start=1):
                # Reciprocal rank component
                rrf = 1.0 / (rank + k)
                # Final contribution combines reciprocal rank + similarity score
                fused_scores[resume_id] += weight * (rank_weight * rrf + (1 - rank_weight) * similarity_score)
    
        # Sort the final combined scores in descending order
        return dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))
        
    def retrieve_resumes(self, job_description: str, top_k: int = 5):
        """
        Retrieve top resumes for a given job description.
    
        - Advanced mode: generates/scored subqueries and fuses results with score-aware RRF.
        - Basic mode: retrieves directly from the FAISS engine.
    
        Args:
            job_description (str): The job description.
            top_k (int): Number of resumes to return. Defaults to 5.
    
        Returns:
            List[str]: Top-K resume IDs.
        """
        if self.rag_mode == "basic":
            # Basic mode:
            # retrieve directly from the FAISS engine using the full job description
            return self.resume_engine.retrieve_top_k(job_description, top_k)
    
        # Advanced mode:
        # 1. Get subqueries from cache or generate them via LLM
        subqueries = self._get_or_generate_subqueries(job_description)
    
        if not subqueries:
            # Fallback to using the original job description as a single subquery with default weight
            scored_subqueries = [(job_description, 1.0)]
        else:
            # 2. Score subqueries using optimizer
            scored_subqueries = self.subquery_optimizer.score_subqueries(job_description, subqueries)
    
            # 3. Insert the original job description with a small boost
            best_score = scored_subqueries[0][1] if scored_subqueries else 1.0
            scored_subqueries.insert(0, (job_description, max(1.2, best_score)))
    
        # 4. Retrieve top resumes per subquery (with similarity scores), and prepare input for fusion
        subquery_info = []
        for subquery_text, optimizer_weight in scored_subqueries:
            # 4.1. Retrieve resume scores using the FAISS engine
            resume_scores = self.resume_engine.retrieve_top_k(subquery_text, top_k, True)
    
            # 4.2. Store each subquery’s metadata and results
            subquery_info.append({
                "weight": optimizer_weight,    # Weight for this subquery
                "resume_scores": resume_scores # {resume_id: score}
            })
    
        # 5. Apply score-aware Reciprocal Rank Fusion to aggregate results across subqueries
        fused_scores = self.score_aware_fusion(subquery_info)
    
        # 6. Return only the top-K resume IDs after fusion
        return list(fused_scores)[:top_k]
            
    def run(self, test_csv_path: str, output_csv_path: str, top_k: int = 5):
        """
        Executes the full RAG pipeline on a test dataset of job descriptions.

        Args:
            test_csv_path (str): Path to CSV containing job descriptions and ground truth labels.
            output_csv_path (str): Path to save the final output CSV with generated answers.
            top_k (int): Number of top resumes to retrieve per job description (default: 5).
        """
        # 1. Load test cases from CSV
        df = pd.read_csv(test_csv_path)
    
        # Clear any previous CUDA allocations
        torch.cuda.empty_cache()
    
        output_rows = []
    
        # 2. Process each job description row
        for row_id, row in df.iterrows():
            job_description = row["Job Description"]
            ground_truth    = row["Ground Truth"]
    
            # 2.1. Retrieve top-K resume IDs (basic or advanced mode)
            top_resume_ids = self.retrieve_resumes(job_description, top_k)
    
            # 2.2. Fetch and join full resumes as a single input context block
            combined_resumes = "==============".join(self.resume_engine.get_resumes_by_ids(top_resume_ids))

            # 2.3. Generate contextual summary using the LLaMA model
            generated_summary = self.summary_model.generate_response(job_description, combined_resumes)
    
            logging.info(f"{row_id}: ✅ Response generation complete.")
    
            # 2.4. Append the result
            output_rows.append({
                "question":     job_description,
                "ground_truth": ground_truth,
                "answer":       generated_summary,
                "contexts":     combined_resumes
            })
    
            # 2.5. Free memory
            gc.collect()
            torch.cuda.empty_cache()
    
        # 3: Save all rows to output CSV
        pd.DataFrame(output_rows).to_csv(output_csv_path, index=False)