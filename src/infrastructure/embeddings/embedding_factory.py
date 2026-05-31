from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config.settings import Settings

class EmbeddingFactory:
    """Factory to initialize and retrieve LangChain Embedding provider adapters."""
    
    @staticmethod
    def get_provider() -> Embeddings:
        """Initializes and returns the configured default LangChain Embeddings provider."""
        provider_type = Settings.EMBEDDING_PROVIDER
        if provider_type == "openai":
            return EmbeddingFactory.get_openai_provider()
        elif provider_type == "local":
            return EmbeddingFactory.get_minilm_provider()
        else:
            raise ValueError(f"Unknown embedding provider: {provider_type}")

    @staticmethod
    def get_minilm_provider() -> Embeddings:
        """Returns the local baseline sentence-transformers/all-MiniLM-L6-v2 provider (384 dim)."""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    @staticmethod
    def get_bge_provider() -> Embeddings:
        """Returns the local advanced BAAI/bge-small-en-v1.5 provider (384 dim)."""
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

    @staticmethod
    def get_openai_provider(api_key: str = None) -> Embeddings:
        """Returns the cloud production standard OpenAI text-embedding-3-small provider (1536 dim)."""
        key = api_key or Settings.OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY is required to initialize the OpenAI embedding provider.")
        return OpenAIEmbeddings(
            openai_api_key=key,
            model="text-embedding-3-small"
        )
