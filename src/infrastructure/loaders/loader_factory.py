import os
from typing import Dict, Type
from src.infrastructure.loaders.base_loader import BaseDocumentLoader
from src.infrastructure.loaders.pdf_loader import PDFLoader
from src.infrastructure.loaders.docx_loader import DOCXLoader
from src.utils.logger import logger

class DocumentLoaderFactory:
    """Factory to resolve the appropriate BaseDocumentLoader strategy dynamically based on the file format."""
    
    _registry: Dict[str, Type[BaseDocumentLoader]] = {
        ".pdf": PDFLoader,
        ".docx": DOCXLoader
    }
    
    @classmethod
    def get_loader(cls, file_path: str) -> BaseDocumentLoader:
        """Returns an instance of the appropriate BaseDocumentLoader strategy for a given file path.
        
        Raises:
            ValueError: If the file extension is not registered in the registry.
        """
        _, ext = os.path.splitext(file_path.lower())
        loader_class = cls._registry.get(ext)
        
        if not loader_class:
            logger.error(f"Unsupported file format '{ext}' for file: {file_path}")
            raise ValueError(
                f"Unsupported file format: '{ext}'. "
                f"Currently supported formats: {', '.join(cls._registry.keys())}"
            )
            
        logger.info(f"Resolved ingestion strategy '{loader_class.__name__}' for file: {file_path}")
        return loader_class()
