import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Settings:
    """Application configuration and file system path resolution."""
    # settings.py is inside src/config/, so parent.parent.parent points to root SDK/
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    # Environment configs
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    
    # Data Paths
    QDRANT_STORAGE_DIR = os.getenv("QDRANT_STORAGE_DIR", str(BASE_DIR / ".qdrant_data"))
    
    @classmethod
    def validate(cls):
        """Validates configuration sanity."""
        if cls.EMBEDDING_PROVIDER not in ["local", "openai"]:
            raise ValueError(f"Invalid EMBEDDING_PROVIDER '{cls.EMBEDDING_PROVIDER}'. Must be 'local' or 'openai'.")
        
        if cls.EMBEDDING_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER is set to 'openai'.")
