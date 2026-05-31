from dataclasses import dataclass

@dataclass
class Chunk:
    """Represents a single semantic snippet extracted from a parent Document with rich schema versioning."""
    id: str
    document_id: str
    notebook_id: str
    text: str
    page_number: int  # 1-indexed; 0 for non-paginated files
    chunk_index: int
    token_count: int = 0
    embedding_model: str = "all-MiniLM-L6-v2"
    schema_version: str = "v1.0.0"
