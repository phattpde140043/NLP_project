import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LCDocument
from src.infrastructure.loaders.base_loader import BaseDocumentLoader
from src.domain.exceptions import CorruptedFileError, SecurityValidationError
from src.utils.security import validate_file_security
from src.utils.logger import logger

class PDFLoader(BaseDocumentLoader):
    """Loads PDF documents using PyPDFLoader under the Ingestion Strategy pattern with robust validation."""
    
    def load(self, file_path: str) -> List[LCDocument]:
        """Parses a PDF using PyPDFLoader after performing binary security validation.
        
        Raises:
            SecurityValidationError: If the PDF signature check fails.
            CorruptedFileError: If the PDF structure is corrupted or encrypted.
        """
        logger.info(f"PDFLoader: Running security validation for: {file_path}")
        # 1. Run security check (size, Magic Bytes)
        validate_file_security(file_path)
        
        # 2. Attempt parsing
        logger.info(f"PDFLoader: Parsing PDF file content: {file_path}")
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"PDFLoader parsing failed: {str(e)}")
            
            # Check for encryption/decryption indicators
            if "encrypted" in error_str or "password" in error_str or "decrypt" in error_str:
                raise SecurityValidationError(
                    "Tài liệu PDF đã bị mã hóa hoặc bảo vệ bằng mật khẩu. "
                    "Vui lòng gỡ mật khẩu trước khi tải lên hệ thống."
                )
            
            raise CorruptedFileError(
                f"Không thể đọc cấu trúc nhị phân của tệp PDF này. Tệp tin có thể bị hỏng (corrupted) "
                f"hoặc chứa bảng mã ký tự không hợp lệ. Chi tiết: {str(e)}"
            )
