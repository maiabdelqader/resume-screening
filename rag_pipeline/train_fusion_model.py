"""
train_fusion_model.py

This script trains a lightweight data-driven model that learns how to fuse dense
(FAISS) and sparse (TF-IDF) similarity scores during resume retrieval. Instead
of the earlier fixed 0.6/0.4 rule, we fit a logistic regression classifier using
pseudo-labels derived from that heuristic.

Key points:
- Builds a pseudo-labeled dataset using the 0.6/0.4 fusion rule.
- Extracts top-K chunks per job description and collects semantic + keyword scores.
- Uses the heuristic to determine which resume should be considered relevant.
- Trains a logistic regression model to approximate this behaviour.
- Saves the trained model for use inside ResumeEngine.
"""

import os
from dotenv import load_dotenv

import joblib
import numpy  as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from resume_engine import ResumeEngine

# Load environment variables (FAISS path, resumes path, embedding model)
load_dotenv("../.env")


def build_training_dataset(resume_engine: ResumeEngine, train_paths, chunk_limit: int = 50):
    """
    Builds a pseudo-labeled dataset for training the fusion model.

    Args:
        resume_engine (ResumeEngine): Provides FAISS + TF-IDF scoring.
        train_paths (List[str]): Paths to CSV files used for pseudo-supervision.
        chunk_limit (int): Number of FAISS chunks to retrieve per query.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix X and label vector y.
    """
    X, y = [], []

    for path in train_paths:
        print(f"Loading training file: {path}")
        df = pd.read_csv(path)

        for _, row in df.iterrows():
            job_description = str(row["Job Description"]).strip()

            # 1. Retrieve top-K semantic candidates from FAISS
            faiss_results = resume_engine.faiss_index.similarity_search_with_score(
                f"query: {job_description}", k=chunk_limit
            )
            if not faiss_results:
                continue  # skip rare cases

            chunk_texts = [doc.page_content for doc, _ in faiss_results]
            semantic_raw = [score for _, score in faiss_results]

            # 2. Compute TF-IDF keyword similarity
            try:
                query_vec = resume_engine.tfidf_vectorizer.transform([job_description])
                chunk_vecs = resume_engine.tfidf_vectorizer.transform(chunk_texts)
                keyword_raw = (query_vec @ chunk_vecs.T).toarray().flatten()
            except Exception:
                keyword_raw = np.zeros(len(chunk_texts))

            # 3. Normalize score arrays
            def normalize(values, invert=False):
                values = np.array(values)
                if len(values) == 0:
                    return values

                min_val = values.min()
                max_val = values.max()

                if max_val == min_val:
                    norm = np.ones_like(values)
                else:
                    norm = (values - min_val) / (max_val - min_val)

                return 1.0 - norm if invert else norm

            semantic_norm = normalize(semantic_raw, invert=True)   # FAISS distance → similarity
            keyword_norm  = normalize(keyword_raw)

            # 4. Apply heuristic fusion rule to determine the best resume
            fusion_scores = 0.6 * semantic_norm + 0.4 * keyword_norm
            best_idx = int(np.argmax(fusion_scores))

            best_doc, _ = faiss_results[best_idx]
            best_resume_id = str(best_doc.metadata.get("ID", "UNKNOWN")).strip()

            # 5. Label chunks (1 = belongs to best resume)
            for (doc, _), s_score, k_score in zip(faiss_results, semantic_norm, keyword_norm):
                resume_id = str(doc.metadata.get("ID", "UNKNOWN")).strip()
                label = 1 if resume_id == best_resume_id else 0
                X.append([s_score, k_score])
                y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"Training samples: {X.shape[0]}   positives: {int(y.sum())}")
    return X, y


def main():
    """
    Main routine for training the fusion model.

    Steps:
    1. Load configuration and initialize ResumeEngine.
    2. Build pseudo-labeled dataset using selected test sets.
    3. Fit a logistic regression model.
    4. Compute AUC on training data (diagnostic).
    5. Save the trained model for resume_engine.py to use at inference time.
    """
    faiss_path      = os.getenv("FAISS_PATH")
    resumes_path    = os.getenv("RESUMES_PATH")
    embedding_model = os.getenv("EMBEDDING_MODEL")

    # 1. Initialize ResumeEngine (loads FAISS + TF-IDF vectorizer)
    resume_engine = ResumeEngine(
        model_name=embedding_model,
        faiss_path=faiss_path,
        resumes_path=resumes_path
    )

    # 2. Select CSVs used for supervision
    train_paths = [
        "../data/test-sets/testset-1.csv",
        "../data/test-sets/testset-2.csv",
    ]

    print("Building training dataset...")
    X, y = build_training_dataset(resume_engine, train_paths, chunk_limit=50)
    print(f"Training samples: {X.shape[0]} | positives: {y.sum()}")

    if X.size == 0:
        print("No data available for training.")
        return

    # 3. Train logistic regression fusion model
    print("Training logistic-regression fusion model...")
    fusion_model = LogisticRegression(
        solver="liblinear",
        l1_ratio=0.0,
        C=1.0,
        max_iter=1000,
        class_weight="balanced"
    )
    fusion_model.fit(X, y)

    # 4. Quick training AUC (not a proper evaluation)
    y_prob = fusion_model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    print(f"Training AUC: {auc:.3f}")

    # 5. Save model for inference
    os.makedirs("../models", exist_ok=True)
    fusion_model_path = "../models/dense_sparse_fusion_lr.joblib"
    joblib.dump(fusion_model, fusion_model_path)

    print(f"Saved fusion model → {fusion_model_path}")


if __name__ == "__main__":
    main()
