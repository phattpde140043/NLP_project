import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from src.config import Config
from src.models import Chunk
from src.utils.logger import logger

class VectorStorageService:
    """Manages vector database storage and semantic retrieval using LangChain Qdrant Vector Store."""
    
    COLLECTION_NAME = "notebook_documents"
    
    def __init__(self, embedding_provider: Embeddings, storage_dir: str = Config.QDRANT_STORAGE_DIR):
        self.embedding_provider = embedding_provider
        self.storage_dir = storage_dir
        
        # Ensure directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # 1. Initialize local embedded Qdrant client
        logger.info(f"Connecting to embedded Qdrant database at: {self.storage_dir}")
        self.client = QdrantClient(path=self.storage_dir)
        
        # 2. Wrap Qdrant client inside LangChain's QdrantVectorStore
        logger.info(f"Wrapping in LangChain QdrantVectorStore (Cosine distance)...")
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.COLLECTION_NAME,
            embedding=self.embedding_provider
        )

    def store_chunks(self, chunks: List[Chunk]):
        """Converts and stores Chunk models into LangChain's VectorStore (Qdrant)."""
        if not chunks:
            logger.warning("No chunks provided to store_chunks.")
            return
            
        # Convert Chunk domain models to LangChain core Documents
        lc_docs = []
        for chunk in chunks:
            lc_docs.append(
                LCDocument(
                    page_content=chunk.text,
                    metadata={
                        "notebook_id": chunk.notebook_id,
                        "document_id": chunk.document_id,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index
                    }
                )
            )
            
        logger.info(f"Storing {len(lc_docs)} document chunks into LangChain Vector Store...")
        # add_documents automatically handles embedding generation via the provider
        self.vector_store.add_documents(documents=lc_docs)
        logger.info("LangChain Vector Store indexing complete.")

    def search_similar_chunks(
        self, 
        notebook_id: str, 
        query_text: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Performs LangChain-based isolated similarity search over a notebook's chunks."""
        logger.info(f"Executing LangChain VectorStore similarity search for Notebook: {notebook_id}")
        
        # Define isolated metadata filter on notebook_id
        notebook_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.notebook_id",  # LangChain nests metadata under 'metadata' field in Qdrant payload
                    match=models.MatchValue(value=notebook_id)
                )
            ]
        )
        
        # Search using LangChain Qdrant integration
        results = self.vector_store.similarity_search_with_score(
            query=query_text,
            k=limit,
            filter=notebook_filter
        )
        
        # Map back to standard structure
        search_results = []
        for doc, score in results:
            metadata = doc.metadata
            search_results.append({
                "score": score,
                "text": doc.page_content,
                "page_number": metadata.get("page_number", 0),
                "document_id": metadata.get("document_id", ""),
                "chunk_index": metadata.get("chunk_index", 0)
            })
            
        return search_results

    def delete_document_chunks(self, document_id: str):
        """Deletes all vector chunks associated with a specific document."""
        logger.info(f"Deleting all vector chunks in Qdrant for Document {document_id}")
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )

    def delete_notebook_chunks(self, notebook_id: str):
        """Deletes all vector chunks associated with a specific notebook."""
        logger.info(f"Deleting all vector chunks in Qdrant for Notebook {notebook_id}")
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.notebook_id",
                            match=models.MatchValue(value=notebook_id)
                        )
                    ]
                )
            )
        )
