import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from src.config.settings import Settings
from src.domain.models.chunk import Chunk
from src.utils.logger import logger

class QdrantStoreService:
    """Infrastructure adapter for local Qdrant Vector database managing multi-model collections."""
    
    COLLECTION_MINILM = "nlp_workspace_documents_minilm"
    COLLECTION_BGE = "nlp_workspace_documents_bge"
    COLLECTION_OPENAI = "nlp_workspace_documents_openai"
    
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or Settings.QDRANT_STORAGE_DIR
        
        # Ensure database directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # 1. Connect to local embedded Qdrant Client
        logger.info(f"Connecting to embedded Qdrant client at: {self.storage_dir}")
        self.client = QdrantClient(path=self.storage_dir)
        
        # 2. Initialize Local MiniLM Baseline Collection (dim=384, M=16)
        self._init_collection(
            collection_name=self.COLLECTION_MINILM,
            vector_size=384,
            m=16,
            ef_construct=100
        )
        
        # 3. Initialize Local BGE-Small Advanced Collection (dim=384, M=32)
        self._init_collection(
            collection_name=self.COLLECTION_BGE,
            vector_size=384,
            m=32,
            ef_construct=200
        )
        
        # 4. Initialize Cloud OpenAI Collection if API key is present at startup
        if Settings.OPENAI_API_KEY:
            self._init_collection(
                collection_name=self.COLLECTION_OPENAI,
                vector_size=1536,
                m=32,
                ef_construct=200
            )

    def _init_collection(self, collection_name: str, vector_size: int, m: int, ef_construct: int):
        """Helper to create a Qdrant collection with customized HNSW parameters and payload pre-indexing."""
        if not self.client.collection_exists(collection_name):
            logger.info(f"Creating collection '{collection_name}' explicitly (dim={vector_size}, HNSW: M={m}, ef_construct={ef_construct})")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=m,
                    ef_construct=ef_construct,
                    on_disk=False
                )
            )
            
            # Create pre-filtering payload index on notebook_id (workspace UUID)
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="metadata.notebook_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
            except Exception as e:
                logger.warning(f"Could not create payload index on {collection_name}: {str(e)}")

    def ensure_openai_collection(self, api_key: str = None):
        """Ensures that the OpenAI collection is initialized when the API key is dynamically supplied by the user."""
        key = api_key or Settings.OPENAI_API_KEY
        if key:
            self._init_collection(
                collection_name=self.COLLECTION_OPENAI,
                vector_size=1536,
                m=32,
                ef_construct=200
            )

    def store_chunks(self, chunks: List[Chunk], collection_name: str, embedding_provider: Embeddings):
        """Indexes and saves Domain Chunk entities into a specific Qdrant Vector Store collection."""
        if not chunks:
            logger.warning("No chunks to index.")
            return
            
        lc_docs = []
        for chunk in chunks:
            lc_docs.append(
                LCDocument(
                    page_content=chunk.text,
                    metadata={
                        "notebook_id": chunk.notebook_id,
                        "document_id": chunk.document_id,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                        "embedding_model": chunk.embedding_model,
                        "schema_version": chunk.schema_version
                    }
                )
            )
            
        logger.info(f"Indexing {len(lc_docs)} chunks in Qdrant collection: {collection_name}...")
        
        # Instantiate a dynamic LangChain QdrantVectorStore adapter targeting this collection
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embedding_provider
        )
        vector_store.add_documents(documents=lc_docs)
        logger.info(f"Indexing in {collection_name} completed successfully.")

    def search_similar_chunks(
        self, 
        workspace_id: str, 
        query_text: str, 
        collection_name: str,
        embedding_provider: Embeddings,
        top_k: int = 4,
        similarity_threshold: float = 0.35,
        hnsw_ef_search: int = None
    ) -> List[Dict[str, Any]]:
        """Performs localized similarity search over a specific collection using pre-filtering and custom SearchParams."""
        logger.info(f"Querying Qdrant collection: {collection_name} for Workspace: {workspace_id} (top_k={top_k}, threshold={similarity_threshold}, ef_search={hnsw_ef_search})")
        
        # Enforce strict namespace workspace isolation using Qdrant Payload Filter
        workspace_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.notebook_id",
                    match=models.MatchValue(value=workspace_id)
                )
            ]
        )
        
        # Set custom SearchParams (e.g. ef_search for high-accuracy path)
        search_params = None
        if hnsw_ef_search:
            search_params = models.SearchParams(hnsw_ef=hnsw_ef_search)
            
        # Instantiate dynamic QdrantVectorStore adapter targeting this collection
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embedding_provider
        )
        
        # Search using LangChain Qdrant integration
        results = vector_store.similarity_search_with_score(
            query=query_text,
            k=top_k,
            filter=workspace_filter,
            search_params=search_params
        )
        
        # Parse, filter threshold, and format output
        search_results = []
        for doc, score in results:
            # Score check: enforce minimal similarity confidence threshold
            if score < similarity_threshold:
                continue
                
            metadata = doc.metadata or {}
            
            search_results.append({
                "score": score,
                "text": doc.page_content or "",
                "page_number": metadata.get("page_number", 0),
                "document_id": metadata.get("document_id", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "token_count": metadata.get("token_count", 0),
                "embedding_model": metadata.get("embedding_model", ""),
                "schema_version": metadata.get("schema_version", "")
            })
            
        return search_results

    def delete_workspace_chunks(self, workspace_id: str):
        """Cleans out all vector collections associated with a workspace UUID."""
        logger.info(f"Wiping all vector collections associated with Workspace: {workspace_id}")
        
        collections_to_clean = [self.COLLECTION_MINILM, self.COLLECTION_BGE]
        if self.client.collection_exists(self.COLLECTION_OPENAI):
            collections_to_clean.append(self.COLLECTION_OPENAI)
            
        for col in collections_to_clean:
            try:
                self.client.delete(
                    collection_name=col,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="metadata.notebook_id",
                                    match=models.MatchValue(value=workspace_id)
                                )
                            ]
                        )
                    )
                )
            except Exception as e:
                logger.error(f"Error cleaning workspace vectors from collection {col}: {str(e)}")

    def delete_document_chunks(self, workspace_id: str, document_id: str):
        """Cleans out all vector collections associated with a specific document inside a workspace."""
        logger.info(f"Wiping all vectors associated with Document ID: {document_id} in Workspace: {workspace_id}")
        
        collections_to_clean = [self.COLLECTION_MINILM, self.COLLECTION_BGE]
        if self.client.collection_exists(self.COLLECTION_OPENAI):
            collections_to_clean.append(self.COLLECTION_OPENAI)
            
        for col in collections_to_clean:
            try:
                self.client.delete(
                    collection_name=col,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="metadata.notebook_id",
                                    match=models.MatchValue(value=workspace_id)
                                ),
                                models.FieldCondition(
                                    key="metadata.document_id",
                                    match=models.MatchValue(value=document_id)
                                )
                            ]
                        )
                    )
                )
                logger.info(f"Successfully deleted chunks for document {document_id} from collection {col}")
            except Exception as e:
                logger.error(f"Error cleaning document vectors from collection {col}: {str(e)}")


    def get_workspace_vectors(self, workspace_id: str, collection_name: str) -> tuple:
        """Retrieves all point vectors and their document IDs (as labels) for a given workspace."""
        import numpy as np
        
        # Enforce filter by workspace_id
        workspace_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.notebook_id",
                    match=models.MatchValue(value=workspace_id)
                )
            ]
        )
        
        # Scroll through points
        try:
            if not self.client.collection_exists(collection_name):
                # Returns standard 384 dim empty array if collection doesn't exist
                return np.zeros((0, 384)), []
                
            response, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=workspace_filter,
                limit=10000,  # Practical limit for local workshops
                with_vectors=True,
                with_payload=True
            )
            
            vectors = []
            labels = []
            
            for point in response:
                if point.vector is not None:
                    # In Qdrant embedded, point.vector can be a list or dict of vectors
                    # LangChain Qdrant Store might save it under flat or nested format
                    vector_data = point.vector
                    if isinstance(vector_data, dict):
                        # Named vectors fallback if any
                        vector_data = list(vector_data.values())[0]
                    vectors.append(vector_data)
                    
                    # Extract document_id as label
                    payload = point.payload or {}
                    metadata = payload.get("metadata", {})
                    doc_id = metadata.get("document_id", "unknown_doc")
                    labels.append(doc_id)
                    
            if not vectors:
                dim = 1536 if "openai" in collection_name else 384
                return np.zeros((0, dim)), []
                
            return np.array(vectors), labels
        except Exception as e:
            logger.error(f"Error scrolling workspace vectors from {collection_name}: {str(e)}")
            dim = 1536 if "openai" in collection_name else 384
            return np.zeros((0, dim)), []

