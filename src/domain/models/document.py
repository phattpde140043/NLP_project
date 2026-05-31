from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Document:
    """Represents an uploaded document source file in a Workspace with rich performance metrics."""
    id: str
    notebook_id: str  # Matches Streamlit Workspace UUID
    filename: str
    file_path: str
    status: str       # 'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'
    chunk_count: int
    content_hash: str # SHA-256 hash of file content to detect duplicates
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    schema_version: str = "v1.0.0"
