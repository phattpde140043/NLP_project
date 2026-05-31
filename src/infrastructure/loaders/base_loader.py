from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document as LCDocument

class BaseDocumentLoader(ABC):
    """Abstract Base Class for all document loader strategies in the ingestion pipeline."""
    
    @abstractmethod
    def load(self, file_path: str) -> List[LCDocument]:
        """Loads a document from a file path and returns a list of standardized LangChain Documents."""
        pass
