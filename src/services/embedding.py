from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import Config
from src.utils.logger import logger

def get_embedding_provider() -> Embeddings:
    """Factory function returning the configured LangChain Embeddings provider."""
    provider_type = Config.EMBEDDING_PROVIDER
    
    if provider_type == "openai":
        logger.info("Configured LangChain Embedding Provider: OpenAIEmbeddings")
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbeddings.")
        return OpenAIEmbeddings(
            openai_api_key=Config.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
    elif provider_type == "local":
        logger.info("Configured LangChain Embedding Provider: HuggingFaceEmbeddings (all-MiniLM-L6-v2)")
        # Lazy load community imports to avoid slow startup
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")
