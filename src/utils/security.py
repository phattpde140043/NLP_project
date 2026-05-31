import os
from src.domain.exceptions import FileTooLargeError, SecurityValidationError, UnsupportedFileError
from src.utils.logger import logger

# Maximum file size limit: 50MB (approved by user)
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  

def validate_file_security(file_path: str):
    """Executes high-security binary validation checks: file size and Magic Bytes signature verification.
    
    Raises:
        FileTooLargeError: If document size exceeds 50MB limit.
        SecurityValidationError: If MIME/Magic Bytes mismatch occurs (preventing malicious extension renaming).
        UnsupportedFileError: If the extension is unsupported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found to secure-check: {file_path}")
        
    # 1. Enforce Hard File Size Limit
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        logger.error(f"Security Block: File size {size_mb:.2f}MB exceeds the 50MB limit.")
        raise FileTooLargeError(
            f"Kích thước tệp ({size_mb:.2f}MB) vượt quá giới hạn tối đa cho phép là 50MB. "
            f"Vui lòng giảm dung lượng tệp trước khi tải lên."
        )
        
    # 2. Binary Magic Bytes Signature Verification (Preventing executable renaming exploits)
    _, ext = os.path.splitext(file_path.lower())
    
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)  # Read the first 4 bytes
    except Exception as e:
        logger.error(f"Failed to read file header bytes: {str(e)}")
        raise SecurityValidationError(f"Không thể đọc mã kiểm tra chữ ký nhị phân của tệp: {str(e)}")

    if ext == ".pdf":
        # Standard PDF file header signature: %PDF (hex: 25 50 44 46)
        if not header.startswith(b"%PDF"):
            logger.error("Security Block: File extension is .pdf but binary does not start with %PDF signature.")
            raise SecurityValidationError(
                "Cảnh báo bảo mật: Mismatch chữ ký nhị phân! Tệp tin có đuôi mở rộng là '.pdf' "
                "nhưng cấu trúc nội dung thực tế không phải là PDF. Bước nạp bị chặn."
            )
            
    elif ext == ".docx":
        # Standard Microsoft OpenXML/ZIP header signature: PK\x03\x04 (hex: 50 4b 03 04)
        if not header.startswith(b"PK\x03\x04"):
            logger.error("Security Block: File extension is .docx but binary does not start with PK signature.")
            raise SecurityValidationError(
                "Cảnh báo bảo mật: Mismatch chữ ký nhị phân! Tệp tin có đuôi mở rộng là '.docx' "
                "nhưng cấu trúc nội dung thực tế không phải là Word (ZIP archive). Bước nạp bị chặn."
            )
            
    else:
        logger.error(f"Unsupported file format check attempted: {ext}")
        raise UnsupportedFileError(f"Định dạng tệp '{ext}' hiện chưa được đăng ký hỗ trợ bảo mật.")
        
    logger.info(f"Security Checks Passed: size={file_size} bytes, MIME validation success for format: {ext}")
