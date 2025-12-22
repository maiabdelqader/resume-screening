"""
subquery_optimizer.py

Provide subquery optimization for Retrieval-Augmented Generation (RAG) pipelines.

Key Features:
- Scores subqueries using a multi-objective function that balances coverage, redundancy, specificity, and topical diversity.
- Calculates a hybrid coverage score based on semantic similarity to the job description.
- Applies a continuous, threshold-free penalty to manage query redundancy.
- Calculates a nuanced specificity score by filtering stopwords and rewarding shorter queries.
- Promotes topical diversity using unsupervised clustering with a log-scaled penalty.
- Optimizes performance by pre-computing embeddings in a single pass to reduce redundant calls.
"""

import numpy as np

from collections import defaultdict

from typing import List, Tuple, Dict, Any

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


class SubqueryOptimizer:
    def __init__(self, embedding_model, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0) -> None:
        """
        Initialize the optimizer with scoring weights and dependencies.
    
        Args:
            embedding_model: An initialized text embedding model for calculating similarity.
            alpha (float): Weight for the coverage score.
            beta  (float): Weight for the redundancy penalty.
            gamma (float): Weight for the specificity reward.
            delta (float): Weight for the cluster balance penalty.
        """
        self.embedding_model  = embedding_model
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

        # Store the weights for the scoring formula
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta

    def compute_coverage(self, subquery: str, job_description: str) -> float:
        """
        Computes a coverage score based on the semantic similarity between the subquery and the job description.
    
        Args:
            subquery (str): The subquery text to evaluate.
            job_description (str): The full job description text.
    
        Returns:
            float: The semantic coverage score (cosine similarity).
        """
        # Generate embeddings for both subquery and job description
        subquery_vector = self.embedding_model.embed_query(subquery)
        job_desc_vector = self.embedding_model.embed_query(job_description)

        # Compute cosine similarity
        return cosine_similarity([job_desc_vector], [subquery_vector])[0][0]

    def compute_redundancy(self, target_index: int, subquery_list: List[List[float]]) -> float:
        """
        Computes a threshold-free redundancy penalty based on maximum similarity.

        Args:
            target_index (int): The index of the subquery to evaluate.
            subquery_embeddings (List[List[float]]): The pre-computed embeddings of all subqueries.

        Returns:
            float: The final redundancy penalty score.
        """
        # Compute embeddings for all subqueries
        subquery_embeddings = self.embedding_model.embed_documents(subquery_list)

        target_embedding = subquery_embeddings[target_index]

        max_similarity = 0.0

        # Find the single highest similarity against any other query in the list.
        for i, other_embedding in enumerate(subquery_embeddings):
            if i == target_index:
                continue
            
            similarity = cosine_similarity([target_embedding], [other_embedding])[0][0]
            max_similarity = max(max_similarity, similarity)

        # Use an exponential penalty (squaring) to punish high similarity more severely.
        # e.g., a similarity of 0.9 results in a penalty of 0.81.
        return max_similarity ** 2

    def compute_specificity(self, subquery: str) -> float:
        """
        Estimate specificity of a subquery based on its length.

        Args:
            subquery (str): The subquery text.

        Returns:
            float: Specificity score (higher score for more specific terms).
        """
        # Create a list of words, excluding any stopwords
        words = [
                    word for word in subquery.lower().split()
                    if word not in ENGLISH_STOP_WORDS
                ]
    
        if not words:
            return 0.0
    
        # The score is the inverse of the number of meaningful words.
        # A shorter, more precise query gets a higher score.
        return 1.0 / len(words)

    def compute_balance_penalty(self, subquery: str, cluster_map: Dict[str, str],
                                 cluster_counts: Dict[str, int], total_subqueries: int) -> float:
        """
        Compute penalty for overrepresented clusters.

        Args:
            subquery (str): The subquery text.
            cluster_map (Dict[str, str]): Mapping from subquery to its cluster label.
            cluster_counts (Dict[str, int]): Count of subqueries per cluster.
            total_subqueries (int): Total number of subqueries.

        Returns:
            float: Balance penalty score.
        """
        # Get the cluster label assigned to the subquery
        cluster_label = cluster_map.get(subquery)

        if not cluster_label:
            return 0.0
            
        cluster_size = cluster_counts[cluster_label]

        # Ignore very small clusters to avoid penalizing unique queries
        if cluster_size < 3:
            return 0.0

        # Logarithmic penalty is less aggressive on large clusters than a linear one
        return np.log1p(cluster_size) / np.log1p(total_subqueries)

    def cluster_subqueries(self, subquery_list: List[str]) -> Dict[str, str]:
        """
        Cluster subqueries using KMeans on TF-IDF vectors.

        Args:
            subquery_list (List[str]): List of subquery strings.

        Returns:
            Dict[str, str]: Mapping from subquery to its assigned cluster label.
        """
        # Convert subqueries into TF-IDF vectors
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(subquery_list)

        # Dynamically determine the number of clusters
        num_clusters = max(1, min(len(subquery_list), len(set(subquery_list))))

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        # Map subquery to cluster label
        return {
            subquery_list[i]: f"cluster_{cluster_labels[i]}" 
            for i in range(len(subquery_list))
        }

    def prepare_cluster_context(self, subquery_list: List[str]) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        Prepare the clustering context for balance scoring by calculating cluster assignments and sizes.

        Args:
            subquery_list (List[str]): List of subquery strings.

        Returns:
            Tuple[Dict[str, str], Dict[str, int]]: Mapping of subquery to cluster and cluster size counts.
        """
        # Cluster the subqueries
        cluster_map = self.cluster_subqueries(subquery_list)

        # Count frequency of each cluster
        cluster_counts = defaultdict(int)
        for label in cluster_map.values():
            cluster_counts[label] += 1

        return cluster_map, cluster_counts

    def score_subqueries(self, job_description: str, subqueries: List[str]) -> List[Tuple[str, float]]:
        """
        Score and rank each subquery based on multi-objective optimization.

        Args:
            job_description (str): Job description to compare against.
            subqueries (List[str]): List of subqueries to evaluate.

        Returns:
            List[Tuple[str, float]]: Top-N subqueries with their scores.
        """
        if not subqueries:
            return []

        # Cluster context for balance scoring
        cluster_map, cluster_counts = self.prepare_cluster_context(subqueries)

        total_subqueries  = len(subqueries)

        scored_subqueries = []

        # Score each subquery
        for subquery_index, subquery_text in enumerate(subqueries):
            # Coverage: similarity to job description
            coverage_score = self.compute_coverage(subquery_text, job_description)

            # Redundancy: similarity to other subqueries
            redundancy_score = self.compute_redundancy(subquery_index, subqueries)

            # Specificity: shorter is better
            specificity_score = self.compute_specificity(subquery_text)

            # Balance: penalize overused clusters
            balance_penalty = self.compute_balance_penalty(
                subquery_text, cluster_map, cluster_counts, total_subqueries
            )

            # Final weighted score
            total_score = (
                self.alpha * coverage_score -
                self.beta  * redundancy_score +
                self.gamma * specificity_score -
                self.delta * balance_penalty
            )

            # Append result
            scored_subqueries.append((subquery_text, total_score))

        return scored_subqueries