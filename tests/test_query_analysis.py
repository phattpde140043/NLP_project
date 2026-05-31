import os
import time
import uuid
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.config.settings import Settings
from src.domain.services.query_config import QueryControlPlaneConfig


@pytest.fixture(scope="module", autouse=True)
def _patch_settings_for_test():
    import shutil
    original_qdrant = Settings.QDRANT_STORAGE_DIR
    original_openai = Settings.OPENAI_API_KEY
    
    test_qdrant_dir = str(Settings.BASE_DIR / ".qdrant_query_analysis_test")
    Settings.QDRANT_STORAGE_DIR = test_qdrant_dir
    Settings.OPENAI_API_KEY = ""
    
    yield
    
    Settings.QDRANT_STORAGE_DIR = original_qdrant
    Settings.OPENAI_API_KEY = original_openai
    if os.path.exists(test_qdrant_dir):
        shutil.rmtree(test_qdrant_dir, ignore_errors=True)

@pytest.fixture(scope="module")
def control_plane_config():
    return QueryControlPlaneConfig()

@pytest.fixture(scope="module")
def query_analyzer(control_plane_config):
    from src.application.query_analyzer import QueryAnalyzer
    return QueryAnalyzer(control_plane_config)

@pytest.fixture(scope="module")
def query_expander(control_plane_config):
    from src.application.query_expander import QueryExpander
    return QueryExpander(control_plane_config)


# ---------------------------------------------------------------------------
# 1. Tests for QueryAnalyzer (Stage 1 & 3)
# ---------------------------------------------------------------------------

def test_extract_features_technical_exact(query_analyzer):
    # Query loaded with symbols, caps, and technical acronyms
    query = "ERR_CONNECTION_RESET at org.apache.http.impl.client.DefaultHttpClient (version v1.2)"
    features = query_analyzer.extract_features(query)
    
    # Check features: [has_regex, symbol_density, caps_ratio, token_rarity, query_length, lexical_dominance]
    assert features[0] == 1.0  # regex pattern matched
    assert features[1] > 0.05  # symbol density is high
    assert features[2] > 0.1   # caps ratio is high
    assert features[3] > 0.1   # rare tokens present (error, version)
    assert features[5] > 0.0   # technical names present

def test_extract_features_natural_language(query_analyzer):
    query = "what is the general abstract summary of the second quarter project description"
    features = query_analyzer.extract_features(query)
    assert features[0] == 0.0  # no specific regex patterns
    assert features[1] == 0.0  # no symbol density
    assert features[2] == 0.0  # no caps ratio

def test_calibrated_routing_hysteresis(query_analyzer):
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    
    # 1. Exact query
    q1 = "ERR_CONNECTION_RESET at class.method"
    route1 = query_analyzer.route_query(q1, session_id=session_id)
    assert route1["routing_path"] == "exact"
    
    # 2. Similar exact query - should continue exact
    q2 = "Exception in thread main at index"
    route2 = query_analyzer.route_query(q2, session_id=session_id)
    assert route2["routing_path"] == "exact"
    
    # 3. Soft semantic query - hysteresis filters sudden jump
    q3 = "explain the overview of this report"
    route3 = query_analyzer.route_query(q3, session_id=session_id)
    # With EMA beta = 0.7, exact history smooths out natural semantic query to preserve routing stability
    assert "exact" in route3["smoothed_affinities"]

def test_robust_mad_normalization_outliers(query_analyzer):
    # Cosine scores with massive outliers
    scores = [0.99, 0.42, 0.38, 0.35, 0.34, 0.12]
    robust_s = query_analyzer.compute_robust_scores(scores)
    
    # Check bounds
    assert np.all(robust_s >= -4.0)
    assert np.all(robust_s <= 4.0)
    # Check robust normalized median is close to 0
    assert np.isclose(np.median(robust_s), 0.0, atol=1e-2)

def test_diagnose_uncertainty_vector(query_analyzer):
    raw_scores = [0.85, 0.82, 0.81, 0.79, 0.78, 0.77]
    report = query_analyzer.diagnose_uncertainty(raw_scores)
    
    assert "entropy" in report
    assert "top_gap" in report
    assert "uncertainty_vector" in report
    
    # Top gap between 0.85 and 0.82 should be positive
    assert report["top_gap"] >= 0.0
    assert len(report["uncertainty_vector"]) == 5

def test_hnsw_aging_entropy(query_analyzer):
    # Simulated short/monotonous paths (low entropy) vs high variance paths
    paths_monotonous = [5, 5, 5, 6, 5, 5]
    report_bad = query_analyzer.measure_graph_aging(write_count=500, delete_count=250, traversal_lengths=paths_monotonous)
    assert report_bad["trigger_reindex"] is True  # due to high delete/write ops and low traversal diversity

# ---------------------------------------------------------------------------
# 2. Tests for QueryExpander (Stage 5)
# ---------------------------------------------------------------------------

def test_extract_weighted_anchors(query_expander):
    query = "Check users database table tbl_users for ERR_502 connection failures"
    anchors = query_expander.extract_weighted_anchors(query)
    
    # ERR_502 is an error code (W=3.0)
    assert anchors["err_502"] == 3.0
    # tbl_users is a sql table name (W=2.5)
    assert anchors["tbl_users"] == 2.5
    # Standard alphanumeric words have default base weight (W=1.0)
    assert anchors["database"] == 1.0

def test_anchor_retention_rate(query_expander):
    query = "Invalid settings.storage.path config"
    anchors = query_expander.extract_weighted_anchors(query)
    
    # Generated hypothesis preserves config anchor settings.storage.path
    hypothesis_good = "We must override the local settings.storage.path configurations in main file."
    rate_good, missing_good = query_expander.calculate_anchor_retention(anchors, hypothesis_good)
    assert rate_good >= 0.85
    assert not any("settings.storage.path" in m for m in missing_good)
    
    # Hypothesis misses the technical anchor entirely
    hypothesis_bad = "Check if local storage properties are invalid."
    rate_bad, missing_bad = query_expander.calculate_anchor_retention(anchors, hypothesis_bad)
    assert rate_bad < 0.5
    assert "settings.storage.path" in missing_bad

def test_evaluate_hyde_utility(query_expander):
    # High entropy, small gap => should return True
    should_run, utility = query_expander.evaluate_hyde_utility("short query", entropy=4.2, top_gap=0.08)
    assert should_run is True
    assert utility > 0.0

# ---------------------------------------------------------------------------
# 3. Tests for SemanticCacheManager (Stage 7)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(Settings.QDRANT_STORAGE_DIR), reason="Requires active Qdrant storage dir")
def test_semantic_cache_lifecycle():
    from src.infrastructure.vectorstores.qdrant_store import QdrantStoreService
    from src.application.query_cache import SemanticCacheManager
    
    store = QdrantStoreService(storage_dir=str(Settings.BASE_DIR / ".qdrant_cache_test"))
    cache = SemanticCacheManager(store)
    
    workspace_id = f"ws_cache_{uuid.uuid4().hex[:8]}"
    query = "Explain retrieval cache drift theta limits"
    query_vector = np.random.rand(384)
    query_vector = query_vector / np.linalg.norm(query_vector) # unit length
    
    # Prepare cache data
    mock_response = {
        "answer": "This answers cached limits query.",
        "sources": [{"text": "limits summary page 4", "page_number": 4}]
    }
    
    # 1. Populates Cache
    cache.store_cache(
        query=query,
        workspace_id=workspace_id,
        query_vector=query_vector,
        data=mock_response,
        embedding_model="nlp_workspace_documents_minilm"
    )
    
    # 2. Query Cache Layer 1 (Exact match)
    hit = cache.lookup_cache(
        query=query,
        workspace_id=workspace_id,
        query_vector=query_vector,
        embedding_model="nlp_workspace_documents_minilm"
    )
    assert hit is not None
    assert hit["answer"] == mock_response["answer"]
    
    # Clean up test directories
    import shutil
    shutil.rmtree(str(Settings.BASE_DIR / ".qdrant_cache_test"), ignore_errors=True)

# ---------------------------------------------------------------------------
# 4. Integration Tests for Hedged Parallel Retrieval
# ---------------------------------------------------------------------------

def test_parallel_hedged_circuit_breaker():
    # Simulate high dense retrieval latency (e.g. 150ms) to trigger hedged BM25 breaker
    def slow_search(*args, **kwargs):
        time.sleep(0.06)  # 60ms is above the 40ms SLA threshold
        return [{"score": 0.95, "text": "slow vector context"}]
        
    with patch("src.infrastructure.vectorstores.qdrant_store.QdrantStoreService.search_similar_chunks") as mock_search:
        mock_search.side_effect = slow_search
        
        from src.application.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        
        # Mock local BM25 scroll execution
        pipeline._local_bm25_search = MagicMock(return_value=[
            {"score": 0.88, "text": "fast local BM25 context", "page_number": 1, "document_id": "doc1"}
        ])
        
        # Embeddings provider mockup
        mock_provider = MagicMock()
        mock_provider.embed_query.return_value = [0.1] * 384
        with patch("src.infrastructure.embeddings.embedding_factory.EmbeddingFactory.get_minilm_provider", return_value=mock_provider):
            response = pipeline.query_workspace(
                workspace_id="test_ws",
                question="explain the general abstract summary of this workspace",
                routing_path="baseline"
            )
            
            # Verify circuit breaker was triggered
            assert response["diagnostics"]["circuit_broken"] is True
            assert len(response["sources"]) > 0
            assert response["sources"][0]["text"] == "fast local BM25 context"
