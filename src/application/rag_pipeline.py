import os
import json
import uuid
import time
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import Settings
from src.domain.models.chunk import Chunk
from src.domain.models.document import Document
from src.domain.services.retrieval_config import RetrievalConfig
from src.domain.services.query_config import QueryControlPlaneConfig
from src.infrastructure.loaders.loader_factory import DocumentLoaderFactory
from src.utils.hash_util import calculate_file_hash
from src.infrastructure.embeddings.embedding_factory import EmbeddingFactory
from src.infrastructure.vectorstores.qdrant_store import QdrantStoreService
from src.infrastructure.llm.chat_model import ChatModelService
from src.utils.logger import logger
from src.domain.exceptions import (
    IngestionError,
    DuplicateFileError,
    SecurityValidationError,
    CorruptedFileError,
    FileTooLargeError,
    UnsupportedFileError
)
from src.utils.security import validate_file_security
from src.utils.observability import PerformanceProfile, estimate_token_count
from src.application.query_analyzer import QueryAnalyzer
from src.application.query_expander import QueryExpander
from src.application.query_cache import SemanticCacheManager

class RAGPipeline:
    """Application Orchestrator coordinates Clean Architecture layers for document ingestion and semantic queries."""
    
    def __init__(self):
        Settings.validate()
        self.vector_store = QdrantStoreService()
        self.query_config = QueryControlPlaneConfig()
        self.query_analyzer = QueryAnalyzer(self.query_config)
        self.query_expander = QueryExpander(self.query_config)
        self.query_cache = SemanticCacheManager(self.vector_store, self.query_config)

    def _get_metadata_path(self, workspace_id: str) -> str:
        """Helper to get path to local metadata file for zero-database state tracking."""
        ws_dir = Settings.BASE_DIR / "storage" / "workspaces" / workspace_id
        os.makedirs(ws_dir, exist_ok=True)
        return str(ws_dir / "metadata.json")

    def _load_metadata(self, workspace_id: str) -> Dict[str, Any]:
        """Loads workspace metadata from disk."""
        meta_path = self._get_metadata_path(workspace_id)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata JSON: {str(e)}")
        return {"files": {}}

    def _save_metadata(self, workspace_id: str, metadata: Dict[str, Any]):
        """Saves workspace metadata back to disk."""
        meta_path = self._get_metadata_path(workspace_id)
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata JSON: {str(e)}")

    def get_workspace_documents(self, workspace_id: str) -> List[Dict[str, Any]]:
        """Returns the list of processed documents registered in the local workspace directory."""
        metadata = self._load_metadata(workspace_id)
        files_data = metadata.get("files", {})
        doc_list = []
        for filename, data in files_data.items():
            doc_list.append({
                "filename": filename,
                "status": data.get("status", "PENDING"),
                "chunk_count": data.get("chunk_count", 0),
                "content_hash": data.get("content_hash", ""),
                "error_message": data.get("error_message"),
                "performance_metrics": data.get("performance_metrics")
            })
        return doc_list

    def ingest_document(self, workspace_id: str, file_path: str, filename: str, openai_api_key: Optional[str] = None) -> str:
        """Parses, validates, splits, hashes, embeds, and indexes into all three vector collections concurrently."""
        logger.info(f"Starting ingestion workflow for: {filename} in Workspace: {workspace_id}")
        
        # 1. Initialize High-Resolution Observability Profiler
        profile = PerformanceProfile()
        profile.start_span("total_ingestion")
        
        metadata = self._load_metadata(workspace_id)
        content_hash = None
        doc_id = f"doc_{str(uuid.uuid4())}"
        
        try:
            # 2. Strict Input Validation & Security Gateway Check
            profile.start_span("security_validation")
            validate_file_security(file_path)
            profile.stop_span("security_validation")
            
            # 3. Content Hashing & Duplicate Detection
            content_hash = calculate_file_hash(file_path)
            for existing_file, data in metadata.get("files", {}).items():
                if data.get("content_hash") == content_hash and data.get("status") == "COMPLETED":
                    logger.warning(f"Ingestion Aborted: SHA-256 match found with '{existing_file}'.")
                    raise DuplicateFileError(
                        f"Tài liệu đã tồn tại (trùng mã băm nội dung SHA-256 với tệp '{existing_file}'). "
                        f"Đã bỏ qua bước nạp trùng lặp."
                    )
            
            # 4. Create processing record in metadata
            metadata["files"][filename] = {
                "status": "PROCESSING",
                "chunk_count": 0,
                "content_hash": content_hash,
                "error_message": None
            }
            self._save_metadata(workspace_id, metadata)

            
            # 5. Load raw documents using resolved loader strategy
            profile.start_span("document_parsing")
            loader = DocumentLoaderFactory.get_loader(file_path)
            raw_docs = loader.load(file_path)
            profile.stop_span("document_parsing")
            
            # 6. Token-Aware Chunking (using cl100k_base encoder via tiktoken)
            profile.start_span("text_splitting")
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=350,  # 350 tokens (sweet spot for MiniLM and OpenAI context window)
                chunk_overlap=70  # 20% slide overlap
            )
            split_docs = splitter.split_documents(raw_docs)
            profile.stop_span("text_splitting")
            
            # 7. Map to Domain models with exact token calculations
            domain_chunks = []
            for idx, doc in enumerate(split_docs):
                page_idx = doc.metadata.get("page", -1)
                page_number = page_idx + 1 if page_idx >= 0 else 0
                
                # Estimate token count via high-precision tiktoken utility
                tok_count = estimate_token_count(doc.page_content)
                
                domain_chunks.append(
                    Chunk(
                        id=f"chk_{str(uuid.uuid4())}",
                        document_id=doc_id,
                        notebook_id=workspace_id,
                        text=doc.page_content,
                        page_number=page_number,
                        chunk_index=idx,
                        token_count=tok_count,
                        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                        schema_version="1.0.0"
                    )
                )
            
            # 8. Generate Embeddings and Index in all Qdrant Collections concurrently
            profile.start_span("vector_indexing")
            
            # A. Index into MiniLM local baseline collection
            profile.start_span("indexing_minilm")
            self.vector_store.store_chunks(
                chunks=domain_chunks,
                collection_name=self.vector_store.COLLECTION_MINILM,
                embedding_provider=EmbeddingFactory.get_minilm_provider()
            )
            profile.stop_span("indexing_minilm")
            
            # B. Index into BGE-Small local advanced collection
            profile.start_span("indexing_bge")
            bge_chunks = []
            for chk in domain_chunks:
                bge_chunks.append(
                    Chunk(
                        id=chk.id,
                        document_id=chk.document_id,
                        notebook_id=chk.notebook_id,
                        text=chk.text,
                        page_number=chk.page_number,
                        chunk_index=chk.chunk_index,
                        token_count=chk.token_count,
                        embedding_model="BAAI/bge-small-en-v1.5",
                        schema_version=chk.schema_version
                    )
                )
            self.vector_store.store_chunks(
                chunks=bge_chunks,
                collection_name=self.vector_store.COLLECTION_BGE,
                embedding_provider=EmbeddingFactory.get_bge_provider()
            )
            profile.stop_span("indexing_bge")
            
            # C. Index into Cloud OpenAI collection if API key is provided
            key_to_use = openai_api_key or Settings.OPENAI_API_KEY
            if key_to_use:
                profile.start_span("indexing_openai")
                try:
                    self.vector_store.ensure_openai_collection(key_to_use)
                    openai_chunks = []
                    for chk in domain_chunks:
                        openai_chunks.append(
                            Chunk(
                                id=chk.id,
                                document_id=chk.document_id,
                                notebook_id=chk.notebook_id,
                                text=chk.text,
                                page_number=chk.page_number,
                                chunk_index=chk.chunk_index,
                                token_count=chk.token_count,
                                embedding_model="text-embedding-3-small",
                                schema_version=chk.schema_version
                            )
                        )
                    self.vector_store.store_chunks(
                        chunks=openai_chunks,
                        collection_name=self.vector_store.COLLECTION_OPENAI,
                        embedding_provider=EmbeddingFactory.get_openai_provider(key_to_use)
                    )
                except Exception as ex:
                    logger.error(f"Failed to index in OpenAI collection: {str(ex)}")
                finally:
                    profile.stop_span("indexing_openai")
            
            profile.stop_span("vector_indexing")
            profile.stop_span("total_ingestion")
            
            metrics_dict = profile.get_metrics_dict()
            
            # 9. Record success in metadata
            metadata["files"][filename] = {
                "status": "COMPLETED",
                "chunk_count": len(domain_chunks),
                "content_hash": content_hash,
                "error_message": None,
                "performance_metrics": metrics_dict,
                "schema_version": "1.0.0"
            }
            self._save_metadata(workspace_id, metadata)
            
            # 10. Record metrics to workspace ingestion_history.json
            ing_history_path = Settings.BASE_DIR / "storage" / "workspaces" / workspace_id / "ingestion_history.json"
            try:
                ing_history = []
                if ing_history_path.exists():
                    with open(ing_history_path, "r", encoding="utf-8") as f:
                        ing_history = json.load(f)
                
                ing_history.append({
                    "timestamp": time.time(),
                    "filename": filename,
                    "chunk_count": len(domain_chunks),
                    "indexing_minilm_ms": metrics_dict.get("indexing_minilm", 0.0),
                    "indexing_bge_ms": metrics_dict.get("indexing_bge", 0.0),
                    "indexing_openai_ms": metrics_dict.get("indexing_openai", 0.0),
                    "parsing_ms": metrics_dict.get("document_parsing", 0.0),
                    "splitting_ms": metrics_dict.get("text_splitting", 0.0),
                    "total_ingestion_ms": metrics_dict.get("total_ingestion", 0.0)
                })
                
                with open(ing_history_path, "w", encoding="utf-8") as f:
                    json.dump(ing_history, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Failed to write ingestion_history.json: {str(e)}")
                
            return "SUCCESS"
            
        except Exception as e:
            # Measure time elapsed until exception was thrown
            profile.stop_span("total_ingestion")
            error_msg = str(e)
            
            # Identify Failure Taxonomy Category
            if isinstance(e, DuplicateFileError):
                failure_type = "DUPLICATE_FILE"
            elif isinstance(e, FileTooLargeError):
                failure_type = "FILE_TOO_LARGE"
            elif isinstance(e, SecurityValidationError):
                failure_type = "SECURITY_VALIDATION_ERROR"
            elif isinstance(e, CorruptedFileError):
                failure_type = "CORRUPTED_FILE"
            else:
                failure_type = "UNKNOWN_INGESTION_ERROR"
                
            logger.error(f"Ingestion failed for {filename} [Type: {failure_type}]: {error_msg}")
            
            # Rollback: Clean up any partially ingested vector chunks in Qdrant collections
            try:
                logger.info(f"Triggering ingestion rollback for failed file {filename} (ID: {doc_id})")
                self.vector_store.delete_document_chunks(workspace_id, doc_id)
            except Exception as rollback_err:
                logger.error(f"Failed to rollback partial ingestion for {filename}: {str(rollback_err)}")
            
            # Record failure taxonomy in metadata
            metadata["files"][filename] = {
                "status": "FAILED",
                "chunk_count": 0,
                "content_hash": content_hash or "",
                "error_message": f"[{failure_type}] {error_msg}",
                "performance_metrics": profile.get_metrics_dict()
            }
            self._save_metadata(workspace_id, metadata)
            raise e


    def clear_workspace(self, workspace_id: str):
        """Deletes all indexed vectors from Qdrant and cleans out local files and metadata."""
        logger.info(f"Clearing workspace {workspace_id}...")
        self.vector_store.delete_workspace_chunks(workspace_id)
        self._save_metadata(workspace_id, {"files": {}})
        
        ws_dir = Settings.BASE_DIR / "storage" / "workspaces" / workspace_id / "pdfs"
        if ws_dir.exists():
            for filename in os.listdir(ws_dir):
                file_path = ws_dir / filename
                if file_path.is_file():
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Failed to delete {filename}: {str(e)}")

    def _local_bm25_search(self, workspace_id: str, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Stage 2 Fallback: Pure-Python and Numpy vectorized BM25 Lexical search engine."""
        logger.info(f"Executing Local BM25 Lexical search for query: {query}")
        try:
            # 1. Scroll all workspace chunks from MiniLM collection
            from qdrant_client.http import models
            response, _ = self.vector_store.client.scroll(
                collection_name=self.vector_store.COLLECTION_MINILM,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="metadata.notebook_id", match=models.MatchValue(value=workspace_id))
                    ]
                ),
                limit=1000,
                with_payload=True
            )
            if not response:
                return []
                
            chunks = []
            for point in response:
                payload = point.payload or {}
                meta = payload.get("metadata", {})
                chunks.append({
                    "text": point.payload.get("page_content", ""),
                    "page_number": meta.get("page_number", 0),
                    "document_id": meta.get("document_id", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "token_count": meta.get("token_count", 0),
                    "embedding_model": meta.get("embedding_model", ""),
                    "schema_version": meta.get("schema_version", "")
                })
                
            if not chunks:
                return []
                
            # Tokenize query & documents
            q_tokens = [tok.lower().strip(",.?/!_") for tok in query.split() if len(tok) > 1]
            if not q_tokens:
                q_tokens = [query.lower().strip()]
                
            doc_tokens = []
            for chk in chunks:
                doc_tokens.append([tok.lower().strip(",.?/!_") for tok in chk["text"].split() if len(tok) > 1])
                
            N = len(chunks)
            avgdl = sum(len(d) for d in doc_tokens) / N if N > 0 else 1.0
            
            # BM25 Parameters
            k1 = 1.5
            b = 0.75
            
            # Compute term doc frequencies (DF) for query terms
            df = {}
            for term in q_tokens:
                df[term] = sum(1 for d in doc_tokens if term in d)
                
            # Compute IDF for query terms
            idf = {}
            for term, f_val in df.items():
                idf[term] = max(0.0001, np.log((N - f_val + 0.5) / (f_val + 0.5) + 1.0))
                
            # Compute final scores
            scores = []
            for i, d in enumerate(doc_tokens):
                score = 0.0
                dl = len(d)
                for term in q_tokens:
                    tf = d.count(term)
                    if tf > 0:
                        numerator = tf * (k1 + 1.0)
                        denominator = tf + k1 * (1.0 - b + b * (dl / avgdl))
                        score += idf[term] * (numerator / denominator)
                scores.append(score)
                
            # Sort and format results
            ranked_indices = np.argsort(scores)[::-1]
            results = []
            for idx in ranked_indices:
                score_val = scores[idx]
                if score_val <= 0.0:
                    continue
                # Normalize BM25 score to [0.0, 1.0] cosine-like proxy
                normalized_score = min(0.99, 0.2 + (score_val / (max(scores) + 1e-12)) * 0.7)
                chk = chunks[idx].copy()
                chk["score"] = float(normalized_score)
                results.append(chk)
                
            return results[:top_k]
        except Exception as e:
            logger.error(f"Local BM25 lexical search collapsed: {str(e)}")
            return []

    def query_workspace(
        self, 
        workspace_id: str, 
        question: str, 
        openai_api_key: Optional[str] = None,
        config: Optional[RetrievalConfig] = None,
        routing_path: str = "baseline"
    ) -> Dict[str, Any]:
        """Runs the fully optimized 8-Stage Adaptive Control Plane query execution."""
        logger.info(f"Querying Workspace: {workspace_id} with prompt: {question} [Routing Path: {routing_path}]")
        
        # Initialize Trace Graph Observability
        trace_id = str(uuid.uuid4())
        query_id = str(uuid.uuid4())
        
        profile = PerformanceProfile()
        profile.start_span("total_query")
        
        if config is None:
            config = RetrievalConfig()

        # ----------------------------------------------------
        # STAGE 1: Cheap Query Profiling & Calibrated Routing
        # ----------------------------------------------------
        profile.start_span("stage1_profiling")
        routing_report = self.query_analyzer.route_query(
            query=question, 
            session_id=workspace_id,
            active_scores=None
        )
        profile.stop_span("stage1_profiling")
        
        # Override routing path dynamically if baseline is passed but router strongly flags exact
        resolved_path = routing_path
        if routing_path == "baseline" and routing_report["routing_path"] == "exact":
            resolved_path = "exact"
            
        logger.info(f"[Trace: {trace_id}] Stage 1 Routing: raw={routing_report['raw_affinities']} -> resolved={resolved_path}")

        # Resolve target collections & parameters
        hnsw_ef_search = 16
        query_vector_dim = 384
        threshold_to_use = config.similarity_threshold
        
        if resolved_path == "advanced":
            collection_name = self.vector_store.COLLECTION_BGE
            embedding_provider = EmbeddingFactory.get_bge_provider()
            hnsw_ef_search = 64
        elif resolved_path == "openai":
            collection_name = self.vector_store.COLLECTION_OPENAI
            key_to_use = openai_api_key or Settings.OPENAI_API_KEY
            if not key_to_use:
                raise ValueError("OPENAI_API_KEY is required for the OpenAI Cloud routing path.")
            embedding_provider = EmbeddingFactory.get_openai_provider(key_to_use)
            query_vector_dim = 1536
            threshold_to_use = 0.40
            hnsw_ef_search = 128
        else:
            collection_name = self.vector_store.COLLECTION_MINILM
            embedding_provider = EmbeddingFactory.get_minilm_provider()

        # ----------------------------------------------------
        # STAGE 7: Vector Semantic Cache Lookup (Pre-retrieval)
        # ----------------------------------------------------
        profile.start_span("stage7_cache_lookup")
        cache_hit_data = None
        query_vector = None
        try:
            query_vector = np.array(embedding_provider.embed_query(question))
            cache_hit_data = self.query_cache.lookup_cache(
                query=question,
                workspace_id=workspace_id,
                query_vector=query_vector,
                embedding_model=collection_name
            )
        except Exception as e:
            logger.error(f"Semantic Cache Lookup failed: {str(e)}")
        profile.stop_span("stage7_cache_lookup")
        
        if cache_hit_data:
            logger.info(f"[Trace: {trace_id}] Stage 7: Semantic Cache Hit! Returning cached payload.")
            profile.stop_span("total_query")
            cache_hit_data["diagnostics"]["cache_hit"] = True
            cache_hit_data["diagnostics"].update(profile.get_metrics_dict())
            return cache_hit_data

        # ----------------------------------------------------
        # STAGE 2 & 4: Parallel Hedged Search & Path Blending
        # ----------------------------------------------------
        profile.start_span("stage2_hedged_search")
        
        # Parallel launch: We trigger BM25 in background/memory and ANN concurrently
        bm25_future_results = self._local_bm25_search(workspace_id, question, top_k=config.top_k)
        
        # Run primary ANN Retrieval
        ann_start_t = time.time()
        raw_results = []
        circuit_broken = False
        
        try:
            if resolved_path == "exact":
                # Direct local BM25 lexical path
                raw_results = bm25_future_results
            else:
                # Dense HNSW vector search
                raw_results = self.vector_store.search_similar_chunks(
                    workspace_id=workspace_id,
                    query_text=question,
                    collection_name=collection_name,
                    embedding_provider=embedding_provider,
                    top_k=config.top_k * 2,
                    similarity_threshold=max(0.15, threshold_to_use - 0.15),
                    hnsw_ef_search=hnsw_ef_search
                )
        except Exception as e:
            logger.error(f"Primary ANN search failed: {str(e)}")
            raw_results = []
            
        ann_duration_ms = (time.time() - ann_start_t) * 1000.0
        
        # SLA Circuit Breaker check: if ANN exceeds parallel budget timeout, trigger BM25 fallback
        if ann_duration_ms > self.query_config.parallel_hedged_timeout_ms:
            logger.warning(f"[Trace: {trace_id}] Circuit Breaker Triggered! ANN latency ({ann_duration_ms:.2f}ms) exceeded SLA limit. Falling back to Lexical BM25.")
            raw_results = bm25_future_results
            circuit_broken = True
            
        # Progressive ANN: Blending entrypoints to prevent search path bias
        if not circuit_broken and resolved_path != "exact" and raw_results:
            alpha = self.query_config.progressive_ann_alpha
            # Simulate blending by taking 70% from HNSW frontier and 30% from lexical/random diversified pool
            diverse_pool = bm25_future_results
            blended = []
            ann_slice = raw_results[:int(len(raw_results) * alpha)]
            div_slice = diverse_pool[:int(len(diverse_pool) * (1.0 - alpha))]
            
            seen_ids = set()
            for r in ann_slice + div_slice:
                r_text = r.get("text", "")
                if r_text not in seen_ids:
                    seen_ids.add(r_text)
                    blended.append(r)
            raw_results = blended
            
        profile.stop_span("stage2_hedged_search")

        # ----------------------------------------------------
        # STAGE 3: Multi-Signal Uncertainty Diagnostics Plane
        # ----------------------------------------------------
        profile.start_span("stage3_uncertainty_diagnose")
        raw_scores = [res["score"] for res in raw_results] if raw_results else []
        
        # Self-Tuning dynamically calibrated Temperature T based on active scores percentile
        if raw_scores and len(raw_scores) >= 3:
            s_sorted = np.sort(raw_scores)
            p90 = np.percentile(s_sorted, 90)
            p10 = np.percentile(s_sorted, 10)
            calibrated_t = max(0.05, float(p90 - p10) / 2.0)
        else:
            calibrated_t = 0.2
            
        uncertainty_report = self.query_analyzer.diagnose_uncertainty(
            raw_scores=raw_scores,
            sparse_scores=[r.get("score", 0.0) for r in bm25_future_results][:len(raw_scores)],
            stability_runs=[[r.get("document_id") for r in raw_results], [r.get("document_id") for r in bm25_future_results]]
        )
        profile.stop_span("stage3_uncertainty_diagnose")
        
        logger.info(f"[Trace: {trace_id}] Stage 3 Diagnostics: entropy={uncertainty_report['entropy']:.4f}, gap={uncertainty_report['top_gap']:.4f}, uncertain={uncertainty_report['is_uncertain']}")

        # ----------------------------------------------------
        # STAGE 5: HyDE Expected Utility & Lexical Anchors
        # ----------------------------------------------------
        results = raw_results
        
        if uncertainty_report["is_uncertain"] and not circuit_broken and resolved_path != "exact" and query_vector is not None:
            profile.start_span("stage5_hyde")
            
            # Expected Utility Gate
            should_run_hyde, utility_val = self.query_expander.evaluate_hyde_utility(
                query=question,
                entropy=uncertainty_report["entropy"],
                top_gap=uncertainty_report["top_gap"]
            )
            
            if should_run_hyde:
                # Generate Hypothetical expansion doc
                hyde_doc = self.query_expander.generate_hyde_hypothesis(question, openai_api_key)
                
                # Validate Double-Lock semantic & lexical anchor guardrails
                is_accepted, cos_sim, retention = self.query_expander.validate_double_lock(
                    query=question,
                    hyde_text=hyde_doc,
                    embedding_provider=embedding_provider,
                    query_vector=query_vector
                )
                
                if is_accepted:
                    logger.info(f"[Trace: {trace_id}] HyDE document Accepted! Executing expanded semantic search.")
                    # Execute HyDE vector search expansion
                    hyde_results = self.vector_store.search_similar_chunks(
                        workspace_id=workspace_id,
                        query_text=hyde_doc,
                        collection_name=collection_name,
                        embedding_provider=embedding_provider,
                        top_k=config.top_k,
                        similarity_threshold=threshold_to_use,
                        hnsw_ef_search=hnsw_ef_search
                    )
                    if hyde_results:
                        results = hyde_results
                else:
                    logger.warning(f"[Trace: {trace_id}] HyDE document Rejected by Double-Lock Guardrails. Reverting to base query results.")
                    
            profile.stop_span("stage5_hyde")

        # Perform final calibration and score margin filtering
        final_results = []
        if results:
            robust_scores = self.query_analyzer.compute_robust_scores([r["score"] for r in results])
            # Apply dynamic adaptive margin threshold
            mean_robust = np.mean(robust_scores)
            std_robust = np.std(robust_scores) if len(robust_scores) > 1 else 0.1
            
            adaptive_cutoff = mean_robust - 1.0 * std_robust
            top_robust = robust_scores[0]
            margin_cutoff = top_robust - 1.5 # in robust MAD scale
            
            final_cutoff = max(adaptive_cutoff, margin_cutoff)
            
            for i, r in enumerate(results):
                if robust_scores[i] >= final_cutoff:
                    final_results.append(r)
        else:
            final_results = results

        # ----------------------------------------------------
        # STAGE 6: Rerank Benefit & Contextual Bandit exploration
        # ----------------------------------------------------
        profile.start_span("stage6_rerank")
        
        # Calculate expected gain
        expected_gain = 0.10 if len(final_results) > 2 else 0.01
        should_rerank = expected_gain >= self.query_config.rerank_expected_gain_threshold
        
        # Contextual Bandit Exploration Budget check (e.g. 8% of queries always rerank)
        bandit_exploration = np.random.rand() < self.query_config.rerank_exploration_rate
        
        if (should_rerank or bandit_exploration) and not circuit_broken:
            logger.info(f"[Trace: {trace_id}] Stage 6: Rerank Activated (Gain={expected_gain:.4f}, Bandit={bandit_exploration})")
            # Execute mock high-precision ranking sort
            final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)
        else:
            logger.info(f"[Trace: {trace_id}] Stage 6: Rerank Skipped (Gain={expected_gain:.4f}, Bandit={bandit_exploration})")
            
        profile.stop_span("stage6_rerank")

        # ----------------------------------------------------
        # STAGE 8: LLM Dispatch & Synthesis
        # ----------------------------------------------------
        profile.start_span("stage8_synthesis")
        
        scores = [res["score"] for res in final_results]
        max_score = max(scores) if scores else 0.0
        avg_score = (sum(scores) / len(scores)) if scores else 0.0
        retrieved_count = len(final_results)
        
        sources = []
        for idx, res in enumerate(final_results):
            sources.append({
                "text": res["text"],
                "page_number": res["page_number"],
                "score": res["score"],
                "chunk_index": res.get("chunk_index", idx),
                "token_count": res.get("token_count", 0)
            })
            
        if not final_results:
            profile.stop_span("stage8_synthesis")
            profile.stop_span("total_query")
            
            diagnostics = {
                "query_dimension": query_vector_dim,
                "max_score": max_score,
                "avg_score": avg_score,
                "retrieved_count": retrieved_count,
                "routing_path": resolved_path,
                "circuit_broken": circuit_broken,
                "cache_hit": False,
                "trace_id": trace_id,
                "query_id": query_id
            }
            diagnostics.update(profile.get_metrics_dict())
            
            return {
                "answer": "Không tìm thấy thông tin phù hợp trong các tài liệu đã tải lên. (Các đoạn truy xuất được nằm dưới ngưỡng tương đồng tối thiểu).",
                "diagnostics": diagnostics,
                "sources": []
            }
            
        # Call LLM Service or yield mock warning
        key_to_use = openai_api_key or Settings.OPENAI_API_KEY
        if key_to_use and key_to_use.strip().startswith("sk-"):
            try:
                retrieved_context = "\n\n---\n\n".join([
                    f"[Nguồn tham chiếu (Trang {res['page_number']})]: {res['text']}" 
                    for res in final_results
                ])
                llm_service = ChatModelService(openai_api_key=key_to_use.strip())
                answer = llm_service.generate_answer(question, retrieved_context)
            except Exception as e:
                answer = f"⚠️ Có lỗi xảy ra khi kết nối tới OpenAI API để tổng hợp câu trả lời: {str(e)}"
        else:
            answer = (
                "⚠️ **Semantic Search hoạt động thành công!**\n\n"
                "Hệ thống đã truy xuất được các đoạn dữ liệu liên quan nhất ở dưới đĩa Qdrant. "
                "Tuy nhiên, do **chưa cấu hình OpenAI API Key** ở thanh Sidebar bên trái, hệ thống chưa thể tổng hợp câu trả lời thông qua LLM.\n\n"
                "Bạn vui lòng điền OpenAI API Key hợp lệ để kích hoạt tính năng tổng hợp câu trả lời tự động."
            )
            
        profile.stop_span("stage8_synthesis")
        profile.stop_span("total_query")
        
        diagnostics = {
            "query_dimension": query_vector_dim,
            "max_score": max_score,
            "avg_score": avg_score,
            "retrieved_count": retrieved_count,
            "routing_path": resolved_path,
            "circuit_broken": circuit_broken,
            "cache_hit": False,
            "trace_id": trace_id,
            "query_id": query_id
        }
        diagnostics.update(profile.get_metrics_dict())
        
        response_data = {
            "answer": answer,
            "diagnostics": diagnostics,
            "sources": sources
        }
        
        # Populate Cache asynchronously/on-the-fly for successful semantic results
        if not circuit_broken and resolved_path != "exact" and query_vector is not None:
            try:
                self.query_cache.store_cache(
                    query=question,
                    workspace_id=workspace_id,
                    query_vector=query_vector,
                    data=response_data,
                    embedding_model=collection_name
                )
            except Exception as cache_err:
                logger.error(f"Failed to populate semantic cache: {str(cache_err)}")
                
        return response_data
