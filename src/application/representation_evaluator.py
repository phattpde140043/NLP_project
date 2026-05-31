import numpy as np
from typing import List, Dict, Any, Tuple

class RepresentationEvaluator:
    """
    Evaluator for Representation Quality of Embedding Vectors using pure NumPy.
    Protects local-first zero-dependency architecture.
    """
    
    @staticmethod
    def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
        """Computes pairwise cosine similarity matrix for a set of vectors."""
        # Normalize vectors to L2 norm
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-9
        normalized = vectors / norms
        # Dot product of normalized vectors
        return np.dot(normalized, normalized.T)

    @staticmethod
    def calculate_silhouette_score(vectors: np.ndarray, labels: np.ndarray) -> float:
        """
        Computes Silhouette Score using pure NumPy.
        Measures intra-cluster tightness vs inter-cluster separation.
        Range: [-1, 1], where 1 is highly compact/separated.
        """
        n_samples = len(vectors)
        if n_samples <= 1:
            return 0.0
            
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            return 0.0  # Cannot cluster with only one cluster
            
        # Calculate pairwise cosine distances (1 - similarity)
        sim_matrix = RepresentationEvaluator.cosine_similarity_matrix(vectors)
        dist_matrix = 1.0 - sim_matrix
        
        silhouette_vals = []
        
        for i in range(n_samples):
            i_label = labels[i]
            
            # 1. Calculate a(i): Mean distance to other points in the same cluster
            same_cluster_indices = np.where(labels == i_label)[0]
            same_cluster_indices = same_cluster_indices[same_cluster_indices != i]
            
            if len(same_cluster_indices) > 0:
                a_i = np.mean(dist_matrix[i, same_cluster_indices])
            else:
                a_i = 0.0  # Single point cluster
                
            # 2. Calculate b(i): Min mean distance to points in any other cluster
            b_i = float('inf')
            for other_label in unique_labels:
                if other_label == i_label:
                    continue
                other_cluster_indices = np.where(labels == other_label)[0]
                mean_dist_to_other = np.mean(dist_matrix[i, other_cluster_indices])
                b_i = min(b_i, mean_dist_to_other)
                
            # 3. Calculate s(i)
            if a_i == 0.0 and b_i == float('inf'):
                s_i = 0.0
            else:
                s_i = (b_i - a_i) / max(a_i, b_i)
            silhouette_vals.append(s_i)
            
        return float(np.mean(silhouette_vals))

    @staticmethod
    def calculate_isotropy(vectors: np.ndarray) -> float:
        """
        Measures the Isotropy (uniform distribution) of the embedding space.
        Uses the partition function ratio approach (fractional variance of direction).
        Anisotropic spaces collapse into a single narrow cone (isotropy closer to 0).
        Fully isotropic spaces approach 1.
        """
        if len(vectors) < 2:
            return 1.0
            
        # Center the vectors
        centered = vectors - np.mean(vectors, axis=0)
        
        # Singular Value Decomposition (SVD) to get eigenvalues of covariance matrix
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            eigenvalues = s ** 2
            
            # Filter non-zero eigenvalues to handle rank-deficient spaces (when N <= D)
            non_zero = eigenvalues[eigenvalues > 1e-7]
            if len(non_zero) > 0 and np.max(non_zero) > 0:
                # Robust isotropy: ratio of mean non-zero eigenvalue to max eigenvalue
                return float(np.mean(non_zero) / np.max(non_zero))
        except Exception:
            pass
        return 0.5

    @staticmethod
    def calculate_information_density(vectors: np.ndarray, variance_threshold: float = 0.95) -> Dict[str, Any]:
        """
        Analyzes the Dimensional Information Density via SVD/PCA variance.
        Determines how many dimensions are actually carrying informational variance.
        """
        if len(vectors) < 2:
            return {"intrinsic_dimensionality": 1, "explained_variance": [1.0]}
            
        centered = vectors - np.mean(vectors, axis=0)
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            eigenvalues = s ** 2
            total_variance = np.sum(eigenvalues)
            
            if total_variance == 0:
                return {"intrinsic_dimensionality": 1, "explained_variance": []}
                
            explained_variance_ratio = eigenvalues / total_variance
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Find the minimum dimensions needed to capture variance_threshold (e.g. 95%)
            intrinsic_dims = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
            
            return {
                "intrinsic_dimensionality": intrinsic_dims,
                "dimension_utilization_ratio": float(intrinsic_dims / vectors.shape[1]),
                "explained_variance": explained_variance_ratio[:10].tolist()  # Top 10 dims
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def calculate_cosine_drift(original_vector: np.ndarray, perturbed_vector: np.ndarray) -> float:
        """
        Measures stability/drift under perturbation (noise/typos).
        Returns the cosine drift (1.0 - cosine_similarity).
        Smaller is more stable (closer to 0.0).
        """
        norm_orig = np.linalg.norm(original_vector)
        norm_pert = np.linalg.norm(perturbed_vector)
        
        if norm_orig == 0 or norm_pert == 0:
            return 1.0
            
        similarity = np.dot(original_vector, perturbed_vector) / (norm_orig * norm_pert)
        return float(1.0 - similarity)
