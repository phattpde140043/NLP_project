import os
import hashlib
from src.utils.logger import logger

def calculate_file_hash(file_path: str) -> str:
    """Calculates a unique SHA-256 hash of a file for incremental duplicate checks."""
    if not os.path.exists(file_path):
        logger.error(f"File not found to hash: {file_path}")
        raise FileNotFoundError(f"File not found to hash: {file_path}")
    
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate SHA-256 for {file_path}: {str(e)}")
        raise e
