from dataclasses import dataclass

@dataclass
class RetrievalConfig:
    """Configurable parameters for semantic retrieval in Vector Space."""
    top_k: int = 4
    similarity_threshold: float = 0.35  # Standardized threshold for Cosine similarity
    search_type: str = "similarity"     # Supported types: similarity, mmr
