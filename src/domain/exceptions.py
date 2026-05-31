class RAGException(Exception):
    """Base exception class for all RAG system anomalies."""
    pass

class IngestionError(RAGException):
    """Base exception class for all document ingestion workflow failures."""
    pass

class CorruptedFileError(IngestionError):
    """Raised when a document's binary structure is corrupted and unparseable."""
    pass

class UnsupportedFileError(IngestionError):
    """Raised when the resolved file format extension is not in the loader registry."""
    pass

class DuplicateFileError(IngestionError):
    """Raised when a document hash matches an already ingested file in the workspace."""
    pass

class FileTooLargeError(IngestionError):
    """Raised when the document size exceeds the configured payload limit (e.g., 50MB)."""
    pass

class SecurityValidationError(IngestionError):
    """Raised when file signature/MIME header validation fails, detecting malicious renaming/inputs."""
    pass
