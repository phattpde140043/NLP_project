import os
from typing import List
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document as LCDocument
from src.infrastructure.loaders.base_loader import BaseDocumentLoader
from src.domain.exceptions import CorruptedFileError, SecurityValidationError
from src.utils.security import validate_file_security
from src.utils.logger import logger

class DOCXLoader(BaseDocumentLoader):
    """Loads Word (.docx) documents using Docx2txtLoader under the Ingestion Strategy pattern with robust validation."""
    
    def load(self, file_path: str) -> List[LCDocument]:
        """Parses a DOCX using Docx2txtLoader after performing binary security validation.
        
        Raises:
            SecurityValidationError: If the Word signature check fails.
            CorruptedFileError: If the Word zip archive is corrupted.
        """
        logger.info(f"DOCXLoader: Running security validation for: {file_path}")
        # 1. Run security check (size, Magic Bytes)
        validate_file_security(file_path)
        
        # 2. Attempt parsing
        logger.info(f"DOCXLoader: Parsing Word file content: {file_path}")
        try:
            loader = Docx2txtLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"DOCXLoader parsing failed: {str(e)}")
            raise CorruptedFileError(
                f"Không thể đọc cấu trúc tệp Word (.docx) này. Tệp tin có thể bị hỏng (corrupted) "
                f"hoặc chứa cấu trúc lưu trữ ZIP không hợp lệ. Chi tiết: {str(e)}"
            )
