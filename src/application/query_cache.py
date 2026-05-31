import os
import re
import time
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from qdrant_client.http import models
from src.config.settings import Settings
from src.domain.services.query_config import QueryControlPlaneConfig
from src.utils.logger import logger
from src.infrastructure.vectorstores.qdrant_store import QdrantStoreService

class SemanticCacheManager:
    """Stage 7 Control Plane: 2-Stage Semantic Cache, Time/Hit Decay, and Centroid Drift Invalidation."""

    COLLECTION_CACHE = "nlp_semantic_cache"

    def __init__(self, vector_store: QdrantStoreService, config: Optional[QueryControlPlaneConfig] = None):
        self.vector_store = vector_store
        self.client = vector_store.client
        self.config = config or QueryControlPlaneConfig()
        
        # Initialize Cache ANN collection (dim=384, Distance=COSINE)
        if not self.client.collection_exists(self.COLLECTION_CACHE):
            logger.info(f"Initializing dedicated semantic cache collection: {self.COLLECTION_CACHE}")
            self.client.create_collection(
                collection_name=self.COLLECTION_CACHE,
                vectors_config=models.VectorParams(
                    size=384, # MiniLM/BGE query dimension
                    distance=models.Distance.COSINE
                )
            )
            # Add keyword filters for workspace_id and exact fingerprint key
            self.client.create_payload_index(
                collection_name=self.COLLECTION_CACHE,
                field_name="workspace_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.COLLECTION_CACHE,
                field_name="fingerprint",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

    def _canonicalize_query(self, query: str) -> str:
        """Normalizes whitespaces, lowercases, and strips punctuation to ensure cache consistency."""
        cleaned = query.lower().strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"[^\w\s]", "", cleaned)
        return cleaned

    def _get_corpus_epoch(self, workspace_id: str) -> int:
        """Retrieves the current corpus epoch to protect against stale cache entries."""
        meta_path = Settings.BASE_DIR / "storage" / "workspaces" / workspace_id / "metadata.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    # We can use the count of active documents + total chunks or a custom epoch int
                    files = meta.get("files", {})
                    completed_docs = sum(1 for data in files.values() if data.get("status") == "COMPLETED")
                    total_chunks = sum(data.get("chunk_count", 0) for data in files.values())
                    return completed_docs * 1000 + total_chunks
            except Exception as e:
                logger.error(f"Failed to read metadata for cache epoch: {str(e)}")
        return 0

    def calculate_cache_fingerprint(self, query: str, workspace_id: str, embedding_model: str) -> str:
        """Stage 7: Invalidation key incorporating query, workspace, model, corpus epoch, and ACL."""
        q_canon = self._canonicalize_query(query)
        epoch = self._get_corpus_epoch(workspace_id)
        acl_hash = hashlib.md5(workspace_id.encode("utf-8")).hexdigest()[:6]
        
        raw_key = f"{q_canon}||{workspace_id}||{embedding_model}||{epoch}||{acl_hash}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get_corpus_centroid(self, workspace_id: str) -> Tuple[np.ndarray, int]:
        """Calculates current centroid of document vectors inside the active baseline collection."""
        # RetrieveMiniLM vectors for document centroid calculation
        vectors, _ = self.vector_store.get_workspace_vectors(workspace_id, self.vector_store.COLLECTION_MINILM)
        if len(vectors) == 0:
            return np.zeros(384), 0
        
        # Calculate mathematical mean centroid vector using numpy
        centroid = np.mean(vectors, axis=0)
        return centroid, len(vectors)

    def monitor_centroid_drift(self, workspace_id: str) -> float:
        """Measures global centroid drift vector migration to detect semantic shift in corpus."""
        centroid, doc_count = self.get_corpus_centroid(workspace_id)
        if doc_count == 0:
            return 0.0
            
        meta_path = Settings.BASE_DIR / "storage" / "workspaces" / workspace_id / "metadata.json"
        
        try:
            stored_centroid = None
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    stored_list = meta.get("cache_centroid")
                    if stored_list:
                        stored_centroid = np.array(stored_list)
            
            if stored_centroid is None or stored_centroid.shape != centroid.shape:
                # Store initial centroid in workspace metadata
                self._save_centroid_to_metadata(workspace_id, centroid.tolist())
                return 0.0
                
            # Compute Euclidean drift distance: ||μ_t - μ_t-1||_2
            drift_distance = float(np.linalg.norm(centroid - stored_centroid))
            logger.info(f"Corpus Centroid Drift: distance={drift_distance:.4f} (Threshold={self.config.cache_centroid_drift_theta})")
            
            if drift_distance > self.config.cache_centroid_drift_theta:
                logger.warning(f"Semantic Drift Detected! Euclidean distance ({drift_distance:.4f}) exceeds threshold. Triggering local cache invalidation.")
                self.invalidate_workspace_cache(workspace_id)
                # Store new centroid as baseline
                self._save_centroid_to_metadata(workspace_id, centroid.tolist())
                
            return drift_distance
        except Exception as e:
            logger.error(f"Failed to monitor centroid drift: {str(e)}")
            return 0.0

    def _save_centroid_to_metadata(self, workspace_id: str, centroid_list: List[float]):
        """Helper to write new cache centroid to workspace metadata."""
        meta_path = Settings.BASE_DIR / "storage" / "workspaces" / workspace_id / "metadata.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["cache_centroid"] = centroid_list
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Failed to write cache centroid to metadata: {str(e)}")

    def lookup_cache(self, query: str, workspace_id: str, query_vector: np.ndarray, embedding_model: str) -> Optional[Dict[str, Any]]:
        """Stage 7 Cache Lookup: 2-Stage Match (Exact Hash lookup + Fast HNSW Vector match)."""
        fingerprint = self.calculate_cache_fingerprint(query, workspace_id, embedding_model)
        
        # Monitor Centroid Drift first; if drift > theta, invalidation has cleared the cache
        self.monitor_centroid_drift(workspace_id)
        
        # Layer 1: Exact Match Hash Lookup (O(1))
        # Search Qdrant cache collection with exact fingerprint keyword
        exact_filter = models.Filter(
            must=[
                models.FieldCondition(key="fingerprint", match=models.MatchValue(value=fingerprint)),
                models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id))
            ]
        )
        
        try:
            results = self.client.scroll(
                collection_name=self.COLLECTION_CACHE,
                scroll_filter=exact_filter,
                limit=1,
                with_payload=True
            )[0]
            
            if results:
                logger.info("Semantic Cache Layer 1: Exact fingerprint match found!")
                # Record hit metrics in payload and return
                point = results[0]
                self._increment_hit(point.id, point.payload)
                return point.payload["data"]
        except Exception as e:
            logger.error(f"Layer 1 exact cache query failed: {str(e)}")
            
        # Layer 2: Fast HNSW Vector Search
        workspace_filter = models.Filter(
            must=[
                models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id))
            ]
        )
        
        try:
            query_res = self.client.query_points(
                collection_name=self.COLLECTION_CACHE,
                query=query_vector.tolist(),
                query_filter=workspace_filter,
                limit=1
            )
            ann_results = query_res.points
            
            if ann_results:
                hit = ann_results[0]
                similarity = hit.score
                payload = hit.payload or {}
                
                # Check dynamic decay threshold: similarity * decay(t) * log(hits)
                timestamp = payload.get("timestamp", time.time())
                hit_count = payload.get("hit_count", 1)
                
                delta_t = max(0.0, time.time() - timestamp)
                freshness_decay = np.exp(-self.config.cache_lambda_decay * delta_t)
                hit_amplifier = np.log(1.0 + hit_count)
                
                cache_score = similarity * freshness_decay * hit_amplifier
                logger.info(f"Semantic Cache Layer 2: similarity={similarity:.4f}, decay={freshness_decay:.4f}, hits={hit_count}, score={cache_score:.4f} (Threshold=0.35)")
                
                if cache_score >= 0.35:
                    logger.info("Semantic Cache Layer 2: Adaptive score hit accepted!")
                    self._increment_hit(hit.id, payload)
                    return payload["data"]
        except Exception as e:
            logger.error(f"Layer 2 ANN semantic cache lookup failed: {str(e)}")
            
        return None

    def store_cache(self, query: str, workspace_id: str, query_vector: np.ndarray, data: Dict[str, Any], embedding_model: str):
        """Saves query results into Qdrant Cache Collection."""
        fingerprint = self.calculate_cache_fingerprint(query, workspace_id, embedding_model)
        point_id = hashlib.md5(fingerprint.encode("utf-8")).hexdigest()
        
        payload = {
            "fingerprint": fingerprint,
            "workspace_id": workspace_id,
            "query": query,
            "timestamp": time.time(),
            "hit_count": 1,
            "data": data,
            "embedding_model": embedding_model
        }
        
        try:
            self.client.upsert(
                collection_name=self.COLLECTION_CACHE,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=query_vector.tolist(),
                        payload=payload
                    )
                ]
            )
            logger.info("Successfully populated semantic cache with query vector and payload.")
        except Exception as e:
            logger.error(f"Failed to upsert cache record: {str(e)}")

    def _increment_hit(self, point_id: str, current_payload: Dict[str, Any]):
        """Increments cache hit count and updates active timestamp."""
        try:
            current_payload["hit_count"] = current_payload.get("hit_count", 0) + 1
            current_payload["timestamp"] = time.time()
            self.client.upsert(
                collection_name=self.COLLECTION_CACHE,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=self.client.retrieve(self.COLLECTION_CACHE, [point_id])[0].vector,
                        payload=current_payload
                    )
                ]
            )
        except Exception as e:
            logger.error(f"Failed to increment hit metrics: {str(e)}")

    def invalidate_workspace_cache(self, workspace_id: str):
        """Wipes cache shards associated with a specific workspace."""
        logger.info(f"Invalidating cache records for Workspace ID: {workspace_id}")
        try:
            self.client.delete(
                collection_name=self.COLLECTION_CACHE,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="workspace_id",
                                match=models.MatchValue(value=workspace_id)
                            )
                        ]
                    )
                )
            )
        except Exception as e:
            logger.error(f"Cache invalidation execution failed: {str(e)}")
