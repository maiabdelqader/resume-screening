"""
resume_engine.py

Handles resume ingestion, embedding, and FAISS-based semantic search for Retrieval-Augmented Generation (RAG).

Key Features:
- Loads resumes from a CSV file and converts them into structured text documents with IDs preserved as metadata.
- Prepares text for embedding by adding the E5 “passage:” prefix and splitting into overlapping chunks for finer retrieval.
- Builds or loads a local FAISS index using Hugging Face embeddings with cosine similarity.
- Trains a TF-IDF vectorizer on the full corpus to capture keyword relevance.
- Supports hybrid retrieval by blending semantic similarity from FAISS with keyword similarity from TF-IDF, using weighted fusion and deduplication.
- Retrieves top-K results as ranked IDs or as an ID→score mapping.
- Provides helpers to fetch full resume text by applicant ID for downstream use.
"""

import os
import torch
import logging

import joblib

import numpy  as np
import pandas as pd

from typing import List, Dict, Union

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy

from langchain_huggingface import HuggingFaceEmbeddings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResumeEngine:
    def __init__(
        self,
        model_name:   str,
        faiss_path:   str,
        resumes_path: str
    ) -> None:
        """
        Initialize the resume retrieval engine.

        Args:
            model_name   (str): Hugging Face model ID for the embedding model.
            faiss_path   (str): Directory to load/save the FAISS index.
            resumes_path (str): Path to the input CSV file with a 'Resume' column.
        """
        self.model_name   = model_name
        self.faiss_path   = faiss_path
        self.resumes_path = resumes_path

        # Load and convert resumes from CSV into LangChain Document objects
        self.resumes = self._load_resumes()

        # Initialize and Fit the TF-IDF vectorizer on all resumes text
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
        self._fit_tfidf() 

        #  Load a FAISS index if it exists, otherwise build a new one from the resume documents
        self.faiss_index = self._load_or_build_faiss()

        # Optional: Load learned fusion model (logistic regression) if available
        self.fusion_model = None
        # Construct absolute path to the saved fusion model
        fusion_model_path = os.path.join(
            os.path.dirname(__file__),   # current file directory
            "..",                        # parent directory
            "models",                    # models folder
            "dense_sparse_fusion_lr.joblib"
        )
        fusion_model_path = os.path.normpath(fusion_model_path)

        # Attempt to load the stored logistic regression model
        if os.path.exists(fusion_model_path):
            try:
                self.fusion_model = joblib.load(fusion_model_path)
                logging.info(f"🔁 Loaded learned fusion model from {fusion_model_path}")
            except Exception as e:
                logging.info(f"⚠️ Failed to load learned fusion model: {e}")
                self.fusion_model = None
        else:
            logging.info("ℹ️ No learned fusion model found — using heuristic fusion.")

    def _load_resumes(self) -> List[Document]:
        """
        Load and convert resumes from CSV into LangChain Document objects.

        Returns:
            List[Document]: Parsed documents with resume text and optional metadata.
        """
        df = pd.read_csv(self.resumes_path)

        # Prefix resume content with "passage: " for E5 model compatibility
        df["Resume"] = df["Resume"].apply(lambda x: f"passage: {x.strip()}")

        loader = DataFrameLoader(df, page_content_column="Resume")
        return loader.load()

    def _fit_tfidf(self) -> None:
        """
        Fit the TF-IDF vectorizer on all resume text.
        """
        all_resumes = [doc.page_content for doc in self.resumes]
        self.tfidf_vectorizer.fit(all_resumes)

    def _load_or_build_faiss(self) -> FAISS:
        """
        Load a FAISS index if it exists, otherwise build a new one from the resume documents.

        Returns:
            FAISS: Vector index for semantic search.
        """
        # Initialize the embedding model
        embedder = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": torch.device("cuda")}
        )

        # Check if a pre-built FAISS index already exists on disk
        if os.path.exists(os.path.join(self.faiss_path, "index.faiss")):
            # Load the existing FAISS index from local storage
            return FAISS.load_local(
                folder_path=self.faiss_path,
                embeddings=embedder,
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=True
            )

        # If FAISS index does not exist, build it from resume documents

        # 1. Split resumes into overlapping chunks for finer retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=270)
        text_chunks   = text_splitter.split_documents(self.resumes)

        # 2. Ensure "passage:" prefix for E5-style embedding context
        for chunk in text_chunks:
            chunk.page_content = f"passage: {chunk.page_content.strip()}"

        # 3. Build a FAISS index from the processed document chunks
        faiss_index = FAISS.from_documents(
            text_chunks,
            embedder,
            distance_strategy=DistanceStrategy.COSINE
        )

        # 4. Save the newly built index locally for reuse in future runs
        faiss_index.save_local(self.faiss_path)

        # 5. Return the FAISS index (either loaded or freshly built)
        return faiss_index

    def _retrieve_dense_sparse_fusion(self, query: str, k: int = 5, dense_weight: float = 0.6, sparse_weight: float = 0.4) -> Dict[str, float]:
        """
        Retrieve resumes using dense–sparse fusion (FAISS embeddings + TF-IDF keywords) by combining semantic and keyword similarity.
        
        Retrieves resumes by:
            1. Finding semantically relevant chunks via FAISS.
            2. Scoring by keyword similarity via TF-IDF.  
            3. Fusing scores using weighted average.
            4. Deduplicating to return top-k unique resumes.

        Args:
            query (str): Search query or job description text.
            k (int): Number of resumes to retrieve (default: 5).
            dense_weight  (float): Weight for semantic scores (default: 0.6).
            sparse_weight (float): Weight for keyword scores  (default: 0.4).

        Returns:
            Dict of {resume_id: score} with fused similarity scores
        """
        # 1. Get semantic results using FAISS
        faiss_results = self.faiss_index.similarity_search_with_score(f"query: {query.strip()}", k=k * 20)
        
        # 2. Extract chunk contents and their distance scores
        chunk_contents  = [doc.page_content for doc, _ in faiss_results]
        distance_scores = [score for _, score in faiss_results]
        
        # 3. Calculate keyword similarity for the same chunks
        try:
            query_vector  = self.tfidf_vectorizer.transform([query])
            chunk_vectors = self.tfidf_vectorizer.transform(chunk_contents)
            keyword_similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
        except Exception as e:
            keyword_similarities = np.zeros(len(chunk_contents))
        
        # 4. Normalize both score types to 0-1 range
        def normalize_scores(scores, invert=False):
            if len(scores) == 0:
                return scores
            min_val, max_val = min(scores), max(scores)
            if max_val - min_val == 0:
                return np.ones_like(scores)
            normalized = (scores - min_val) / (max_val - min_val)
            return 1 - normalized if invert else normalized
        
        # Normalize and invert distance scores (lower distance = higher similarity)
        normalized_semantic = normalize_scores(distance_scores, invert=True)
        normalized_keyword  = normalize_scores(keyword_similarities, invert=False)
        
        # 5. Combine semantic + keyword scores:
        #    → Use learned fusion model if available
        #    → Otherwise fall back to the 0.6 / 0.4 heuristic
        if self.fusion_model is not None:
            logging.info("🔁 Using learned fusion model for dense–sparse scoring")
            # Build feature matrix: each chunk → [semantic_norm, keyword_norm]
            feature_matrix = np.column_stack([normalized_semantic, normalized_keyword])

            try:
                # predict_proba returns:
                #   [:,0] → probability of class 0 (non-relevant)
                #   [:,1] → probability of class 1 (relevant)
                # We take p(relevant=1) as the combined score
                combined_scores = self.fusion_model.predict_proba(feature_matrix)[:, 1]
            except Exception as e:
                logging.info(f"⚠️ Fusion model prediction failed, fallback to heuristic: {e}")
                combined_scores = (dense_weight * normalized_semantic) + (sparse_weight * normalized_keyword)
        else:
            logging.info(f"ℹ️ No learned fusion model found, fallback to heuristic")
            # Fallback when no model is available
            combined_scores = (dense_weight * normalized_semantic) + (sparse_weight * normalized_keyword)
        
        # 6. Build scored list of resume chunks
        chunk_candidates = []
        for idx, (doc, _) in enumerate(faiss_results):
            resume_id = doc.metadata.get("ID", f"UNKNOWN_{idx}")
            chunk_candidates.append((idx, combined_scores[idx], resume_id))
        
        # 7. Sort by combined score (descending)
        chunk_candidates.sort(key=lambda x: x[1], reverse=True)

        # 8. Get unique resumes from top-ranked chunks
        seen_resume_ids = set()
        top_resumes = []  # (resume_id, combined_score)
        
        # 9. Deduplicate by resume ID and select top K
        for idx, score, resume_id in chunk_candidates:
            # Skip already-counted resumes
            if resume_id in seen_resume_ids:
                continue

            seen_resume_ids.add(resume_id)
            top_resumes.append((resume_id, score))
        
            # Early exit once we have k unique resumes
            if len(top_resumes) == k:
                break
        
        return dict(top_resumes)

    def retrieve_top_k(self, query: str, k: int = 5, with_scores: bool = False) -> Union[List[str], Dict[str, float]]:
        """
        Retrieve top-K resume IDs using either semantic search (FAISS) or dense–sparse fusion (FAISS + TF-IDF).
    
        Args:
            query (str): The job description or search query to match against resumes.
            k (int): Number of top resumes to retrieve (default: 5).
            with_scores (bool): Whether to return similarity scores along with resume IDs (default: False).
    
        Returns:
            If with_scores=False: 
                List[str] of resume IDs in descending order of similarity
            If with_scores=True: 
                Dict[str, float] mapping resume IDs to similarity scores, where higher scores indicate better matches
        """
        if with_scores:
            # ADVANCED RAG MODE
            # Retrieve resumes using dense–sparse fusion (FAISS + TF-IDF)
            return self._retrieve_dense_sparse_fusion(query, k)
        else:
            # BASIC RAG MODE
            # Use pure semantic search (FAISS)
            docs = self.faiss_index.similarity_search(f"query: {query.strip()}", k=k)
        
            # Extract only resume IDs
            return [doc.metadata.get("ID", "UNKNOWN") for doc in docs]

    def get_resumes_by_ids(self, ids: List[str]) -> List[str]:
        """
        Fetch resume text for a given list of resume IDs.

        Args:
            ids (List[str]): List of resume IDs.

        Returns:
            List[str]: Formatted resume texts prefixed by applicant ID.
        """
        return [
            f"Applicant ID {doc.metadata['ID']}\n{doc.page_content}"
            for doc in self.resumes
            if doc.metadata.get("ID") in ids
        ]
