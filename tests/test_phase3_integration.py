"""
Phase 3 Integration Tests — Multi-Model RAG Evaluation Framework
=================================================================
Validates ingestion profiling, benchmark harness metrics, JSON persistence,
edge-case handling, and performance observability across the 3 embedding paths
(MiniLM / BGE / OpenAI).

All OpenAI-dependent code paths are mocked to allow offline execution.
A dedicated Qdrant storage directory (.qdrant_test_phase3) is used for isolation.
"""

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Test-scoped constants
# ---------------------------------------------------------------------------
SDK_ROOT = Path(__file__).resolve().parent.parent
TEST_QDRANT_DIR = str(SDK_ROOT / ".qdrant_test_phase3")
TEST_WORKSPACE_ID = f"ws_test_{uuid.uuid4().hex[:8]}"
TEST_STORAGE_DIR = SDK_ROOT / "storage" / "workspaces" / TEST_WORKSPACE_ID

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_sample_docx_path() -> str:
    """Returns the path to the sample .docx fixture shipped with the project."""
    path = SDK_ROOT / "sample_docs" / "nlp_test.docx"
    if path.exists():
        return str(path)
    raise FileNotFoundError(
        f"Test fixture not found at {path}. "
        "Ensure sample_docs/nlp_test.docx is present."
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _patch_settings_for_test():
    """
    Module-scoped fixture: patches Settings so that every component in the
    application uses the isolated test Qdrant directory and has no OpenAI key.
    """
    from src.config.settings import Settings

    original_qdrant = Settings.QDRANT_STORAGE_DIR
    original_openai = Settings.OPENAI_API_KEY

    Settings.QDRANT_STORAGE_DIR = TEST_QDRANT_DIR
    Settings.OPENAI_API_KEY = ""

    yield

    # Teardown: restore originals & wipe test artefacts
    Settings.QDRANT_STORAGE_DIR = original_qdrant
    Settings.OPENAI_API_KEY = original_openai

    if os.path.exists(TEST_QDRANT_DIR):
        shutil.rmtree(TEST_QDRANT_DIR, ignore_errors=True)
    if TEST_STORAGE_DIR.exists():
        shutil.rmtree(TEST_STORAGE_DIR, ignore_errors=True)


@pytest.fixture(scope="module")
def rag_pipeline():
    """Module-scoped RAGPipeline instance pointing at the test Qdrant dir."""
    from src.application.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    return pipeline


@pytest.fixture(scope="module")
def ingested_workspace(rag_pipeline):
    """
    Ingests the sample docx into the test workspace once for the entire module.
    Returns the workspace_id so downstream tests can query or benchmark it.
    """
    file_path = _get_sample_docx_path()
    result = rag_pipeline.ingest_document(
        workspace_id=TEST_WORKSPACE_ID,
        file_path=file_path,
        filename="nlp_test.docx",
        openai_api_key=None,
    )
    assert result == "SUCCESS"
    return TEST_WORKSPACE_ID


# ============================================================================
# 1. PerformanceProfile span recording
# ============================================================================


class TestPerformanceProfile:
    """Unit-level tests for the PerformanceProfile observability helper."""

    def test_start_stop_span_returns_positive_duration(self):
        from src.utils.observability import PerformanceProfile

        profile = PerformanceProfile()
        profile.start_span("test_span")
        time.sleep(0.01)  # ~10 ms
        duration = profile.stop_span("test_span")

        assert isinstance(duration, float)
        assert duration > 0.0, "Span duration must be positive"

    def test_stop_span_without_start_returns_zero(self):
        from src.utils.observability import PerformanceProfile

        profile = PerformanceProfile()
        duration = profile.stop_span("nonexistent_span")
        assert duration == 0.0

    def test_get_metrics_dict_excludes_start_markers(self):
        from src.utils.observability import PerformanceProfile

        profile = PerformanceProfile()
        profile.start_span("alpha")
        time.sleep(0.005)
        profile.stop_span("alpha")

        profile.start_span("beta")
        time.sleep(0.005)
        profile.stop_span("beta")

        metrics = profile.get_metrics_dict()
        assert "alpha" in metrics
        assert "beta" in metrics
        # No raw start markers leaked
        assert not any(k.endswith("_start") for k in metrics), (
            "get_metrics_dict must filter out _start markers"
        )

    def test_multiple_spans_are_independent(self):
        from src.utils.observability import PerformanceProfile

        profile = PerformanceProfile()

        profile.start_span("outer")
        profile.start_span("inner")
        time.sleep(0.01)
        inner_dur = profile.stop_span("inner")
        outer_dur = profile.stop_span("outer")

        assert outer_dur >= inner_dur, (
            "Outer span must be >= inner span duration"
        )

    def test_metrics_dict_values_are_numeric(self):
        from src.utils.observability import PerformanceProfile

        profile = PerformanceProfile()
        profile.start_span("x")
        profile.stop_span("x")

        for v in profile.get_metrics_dict().values():
            assert isinstance(v, (int, float))


# ============================================================================
# 2. Ingestion history JSON structure
# ============================================================================


class TestIngestionHistoryStructure:
    """Validates the ingestion_history.json file written after a successful ingestion."""

    REQUIRED_FIELDS = {
        "timestamp",
        "filename",
        "chunk_count",
        "indexing_minilm_ms",
        "indexing_bge_ms",
        "indexing_openai_ms",
        "parsing_ms",
        "splitting_ms",
        "total_ingestion_ms",
    }

    def test_ingestion_history_file_exists(self, ingested_workspace):
        path = TEST_STORAGE_DIR / "ingestion_history.json"
        assert path.exists(), f"ingestion_history.json not found at {path}"

    def test_ingestion_history_is_valid_json_list(self, ingested_workspace):
        path = TEST_STORAGE_DIR / "ingestion_history.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list), "ingestion_history.json must be a JSON array"
        assert len(data) >= 1, "Must have at least one ingestion record"

    def test_ingestion_record_has_required_fields(self, ingested_workspace):
        path = TEST_STORAGE_DIR / "ingestion_history.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        record = data[-1]  # most recent entry
        missing = self.REQUIRED_FIELDS - set(record.keys())
        assert not missing, f"Ingestion record missing fields: {missing}"

    def test_ingestion_record_values_are_correct_types(self, ingested_workspace):
        path = TEST_STORAGE_DIR / "ingestion_history.json"
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)[-1]

        assert isinstance(record["timestamp"], (int, float))
        assert isinstance(record["filename"], str)
        assert isinstance(record["chunk_count"], int)
        assert record["chunk_count"] > 0
        # Timing fields must be non-negative floats
        for timing_field in [
            "indexing_minilm_ms",
            "indexing_bge_ms",
            "total_ingestion_ms",
            "parsing_ms",
            "splitting_ms",
        ]:
            assert isinstance(record[timing_field], (int, float)), (
                f"{timing_field} must be numeric"
            )
            assert record[timing_field] >= 0, (
                f"{timing_field} must be non-negative"
            )

    def test_local_model_indexing_times_are_positive(self, ingested_workspace):
        """MiniLM and BGE indexing must have ran (positive timing), OpenAI can be 0."""
        path = TEST_STORAGE_DIR / "ingestion_history.json"
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)[-1]

        assert record["indexing_minilm_ms"] > 0, "MiniLM indexing time must be > 0"
        assert record["indexing_bge_ms"] > 0, "BGE indexing time must be > 0"
        # OpenAI was not available (no key), so timing should be 0
        assert record["indexing_openai_ms"] == 0, (
            "OpenAI indexing time must be 0 when no API key is provided"
        )


# ============================================================================
# 3. Benchmark harness output structure
# ============================================================================


class TestBenchmarkHarnessOutputStructure:
    """Validates the shape returned by BenchmarkHarness.run_benchmark()."""

    @pytest.fixture(scope="class")
    def benchmark_result(self, rag_pipeline, ingested_workspace):
        from src.application.benchmark_harness import BenchmarkHarness

        harness = BenchmarkHarness(rag_pipeline)
        return harness.run_benchmark(
            workspace_id=ingested_workspace, openai_api_key=None
        )

    def test_top_level_keys(self, benchmark_result):
        assert "aggregated" in benchmark_result
        assert "logs" in benchmark_result

    def test_aggregated_has_three_paths(self, benchmark_result):
        agg = benchmark_result["aggregated"]
        assert isinstance(agg, list)
        assert len(agg) == 3, "Must have exactly 3 path entries (BASELINE, ADVANCED, OPENAI)"

    def test_aggregated_path_names(self, benchmark_result):
        path_names = [
            entry["Định hướng truy hồi (Path)"] for entry in benchmark_result["aggregated"]
        ]
        assert "BASELINE" in path_names
        assert "ADVANCED" in path_names
        assert "OPENAI" in path_names

    def test_aggregated_entries_have_required_metric_keys(self, benchmark_result):
        required_keys = {
            "Định hướng truy hồi (Path)",
            "Độ trễ trung bình (Avg Latency)",
            "Độ phủ ngữ nghĩa (Recall@K)",
            "Độ chính xác (Precision@K)",
            "Thứ hạng nghịch đảo (MRR)",
            "Trạng thái (Status)",
        }
        for entry in benchmark_result["aggregated"]:
            missing = required_keys - set(entry.keys())
            assert not missing, f"Aggregated entry missing keys: {missing}"

    def test_logs_are_list_of_dicts(self, benchmark_result):
        logs = benchmark_result["logs"]
        assert isinstance(logs, list)
        for log_entry in logs:
            assert isinstance(log_entry, dict)

    def test_log_entries_have_expected_fields(self, benchmark_result):
        expected_fields = {
            "question_id",
            "question",
            "path",
            "latency_ms",
            "recall",
            "precision",
            "mrr",
            "chunks_retrieved",
        }
        for log_entry in benchmark_result["logs"]:
            missing = expected_fields - set(log_entry.keys())
            assert not missing, f"Log entry missing fields: {missing}"

    def test_logs_have_entries_for_local_paths(self, benchmark_result):
        paths_in_logs = {e["path"] for e in benchmark_result["logs"]}
        assert "BASELINE" in paths_in_logs, "Logs must include BASELINE runs"
        assert "ADVANCED" in paths_in_logs, "Logs must include ADVANCED runs"


# ============================================================================
# 4. Benchmark harness skips OpenAI without key
# ============================================================================


class TestBenchmarkSkipsOpenAI:
    """When no valid OpenAI key is provided, the OPENAI path must be skipped."""

    @pytest.fixture(scope="class")
    def no_key_result(self, rag_pipeline, ingested_workspace):
        from src.application.benchmark_harness import BenchmarkHarness

        harness = BenchmarkHarness(rag_pipeline)
        return harness.run_benchmark(
            workspace_id=ingested_workspace, openai_api_key=None
        )

    def test_openai_path_has_zero_runs(self, no_key_result):
        openai_entry = next(
            e
            for e in no_key_result["aggregated"]
            if e["Định hướng truy hồi (Path)"] == "OPENAI"
        )
        assert openai_entry["Độ trễ trung bình (Avg Latency)"] == "0.0 ms"
        assert openai_entry["Thứ hạng nghịch đảo (MRR)"] == "0.00"

    def test_no_openai_log_entries(self, no_key_result):
        openai_logs = [e for e in no_key_result["logs"] if e["path"] == "OPENAI"]
        assert len(openai_logs) == 0, "No OPENAI log entries expected without API key"

    def test_openai_status_shows_disabled(self, no_key_result):
        openai_entry = next(
            e
            for e in no_key_result["aggregated"]
            if e["Định hướng truy hồi (Path)"] == "OPENAI"
        )
        assert "VÔ HIỆU" in openai_entry["Trạng thái (Status)"]


# ============================================================================
# 5. Questions.json validity
# ============================================================================


class TestQuestionsJsonValid:
    """Validates the benchmark questions fixture file schema and data integrity."""

    QUESTIONS_PATH = SDK_ROOT / "evaluation" / "questions.json"

    def test_file_exists(self):
        assert self.QUESTIONS_PATH.exists(), "evaluation/questions.json must exist"

    def test_is_valid_json_list(self):
        with open(self.QUESTIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0, "Must have at least one benchmark question"

    def test_question_schema(self):
        with open(self.QUESTIONS_PATH, "r", encoding="utf-8") as f:
            questions = json.load(f)

        required_keys = {"id", "question", "ground_truth_doc"}
        for idx, q in enumerate(questions):
            missing = required_keys - set(q.keys())
            assert not missing, f"Question #{idx} missing keys: {missing}"
            assert isinstance(q["id"], str) and len(q["id"]) > 0
            assert isinstance(q["question"], str) and len(q["question"]) > 0
            assert isinstance(q["ground_truth_doc"], str) and len(q["ground_truth_doc"]) > 0

    def test_question_ids_are_unique(self):
        with open(self.QUESTIONS_PATH, "r", encoding="utf-8") as f:
            questions = json.load(f)
        ids = [q["id"] for q in questions]
        assert len(ids) == len(set(ids)), "Question IDs must be unique"

    def test_expected_page_is_non_negative(self):
        with open(self.QUESTIONS_PATH, "r", encoding="utf-8") as f:
            questions = json.load(f)
        for q in questions:
            if "expected_page" in q:
                assert isinstance(q["expected_page"], int)
                assert q["expected_page"] >= 0


# ============================================================================
# 6. EmbeddingFactory providers
# ============================================================================


class TestEmbeddingFactoryLocalProviders:
    """Verifies that local embedding providers can be instantiated without errors."""

    def test_minilm_provider_instantiation(self):
        from src.infrastructure.embeddings.embedding_factory import EmbeddingFactory

        provider = EmbeddingFactory.get_minilm_provider()
        assert provider is not None

    def test_bge_provider_instantiation(self):
        from src.infrastructure.embeddings.embedding_factory import EmbeddingFactory

        provider = EmbeddingFactory.get_bge_provider()
        assert provider is not None

    def test_minilm_provider_is_embeddings_interface(self):
        from langchain_core.embeddings import Embeddings
        from src.infrastructure.embeddings.embedding_factory import EmbeddingFactory

        provider = EmbeddingFactory.get_minilm_provider()
        assert isinstance(provider, Embeddings)

    def test_bge_provider_is_embeddings_interface(self):
        from langchain_core.embeddings import Embeddings
        from src.infrastructure.embeddings.embedding_factory import EmbeddingFactory

        provider = EmbeddingFactory.get_bge_provider()
        assert isinstance(provider, Embeddings)

    def test_openai_provider_raises_without_key(self):
        from src.infrastructure.embeddings.embedding_factory import EmbeddingFactory

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            EmbeddingFactory.get_openai_provider(api_key=None)

    def test_minilm_produces_384_dim_vector(self):
        from src.infrastructure.embeddings.embedding_factory import EmbeddingFactory

        provider = EmbeddingFactory.get_minilm_provider()
        vectors = provider.embed_documents(["hello world"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 384

    def test_bge_produces_384_dim_vector(self):
        from src.infrastructure.embeddings.embedding_factory import EmbeddingFactory

        provider = EmbeddingFactory.get_bge_provider()
        vectors = provider.embed_documents(["hello world"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 384


# ============================================================================
# 7. Qdrant multi-collection initialization
# ============================================================================


class TestQdrantMultiCollection:
    """Validates that QdrantStoreService initialises MiniLM and BGE collections."""

    def test_collections_created(self, rag_pipeline):
        client = rag_pipeline.vector_store.client
        collections = [c.name for c in client.get_collections().collections]

        assert QdrantStoreService.COLLECTION_MINILM in collections, (
            "MiniLM collection must exist"
        )
        assert QdrantStoreService.COLLECTION_BGE in collections, (
            "BGE collection must exist"
        )

    def test_minilm_collection_has_correct_vector_size(self, rag_pipeline):
        info = rag_pipeline.vector_store.client.get_collection(
            QdrantStoreService.COLLECTION_MINILM
        )
        assert info.config.params.vectors.size == 384

    def test_bge_collection_has_correct_vector_size(self, rag_pipeline):
        info = rag_pipeline.vector_store.client.get_collection(
            QdrantStoreService.COLLECTION_BGE
        )
        assert info.config.params.vectors.size == 384


# ============================================================================
# 8. Ingestion profiling metrics completeness
# ============================================================================


class TestIngestionProfilingMetrics:
    """Verifies that per-file metadata has complete performance_metrics."""

    def test_metadata_file_exists(self, ingested_workspace):
        meta_path = TEST_STORAGE_DIR / "metadata.json"
        assert meta_path.exists()

    def test_metadata_has_completed_status(self, ingested_workspace):
        with open(TEST_STORAGE_DIR / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        file_entry = meta["files"].get("nlp_test.docx")
        assert file_entry is not None, "nlp_test.docx must be in metadata"
        assert file_entry["status"] == "COMPLETED"

    def test_metadata_performance_metrics_present(self, ingested_workspace):
        with open(TEST_STORAGE_DIR / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        file_entry = meta["files"]["nlp_test.docx"]
        perf = file_entry.get("performance_metrics")
        assert perf is not None, "performance_metrics must be set"
        assert isinstance(perf, dict)

    def test_metadata_performance_metrics_has_spans(self, ingested_workspace):
        with open(TEST_STORAGE_DIR / "metadata.json", "r", encoding="utf-8") as f:
            perf = json.load(f)["files"]["nlp_test.docx"]["performance_metrics"]

        expected_spans = [
            "total_ingestion",
            "security_validation",
            "document_parsing",
            "text_splitting",
            "vector_indexing",
            "indexing_minilm",
            "indexing_bge",
        ]
        for span in expected_spans:
            assert span in perf, f"Missing performance span: {span}"
            assert perf[span] >= 0, f"Span {span} must be non-negative"

    def test_total_ingestion_is_greatest_span(self, ingested_workspace):
        with open(TEST_STORAGE_DIR / "metadata.json", "r", encoding="utf-8") as f:
            perf = json.load(f)["files"]["nlp_test.docx"]["performance_metrics"]
        total = perf["total_ingestion"]
        for k, v in perf.items():
            if k != "total_ingestion":
                assert total >= v, (
                    f"total_ingestion ({total}) should be >= {k} ({v})"
                )


# ============================================================================
# 9. Benchmark history JSON persistence
# ============================================================================


class TestBenchmarkHistoryPersistence:
    """Verifies that benchmark_history.json is written by a prior benchmark run.

    This test relies on the TestBenchmarkSkipsOpenAI class-scoped fixture having
    already executed run_benchmark(), which writes to benchmark_history.json.
    """

    def test_benchmark_history_written(self, ingested_workspace):
        path = TEST_STORAGE_DIR / "benchmark_history.json"
        assert path.exists(), "benchmark_history.json must be written after run_benchmark"

    def test_benchmark_history_structure(self, ingested_workspace):
        path = TEST_STORAGE_DIR / "benchmark_history.json"
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)

        assert isinstance(history, list)
        assert len(history) >= 1
        entry = history[-1]
        assert "timestamp" in entry
        assert "aggregated" in entry
        assert "logs" in entry
        assert isinstance(entry["timestamp"], (int, float))


# ============================================================================
# 10. Empty workspace queries
# ============================================================================


class TestEmptyWorkspaceQuery:
    """Validates graceful handling when querying a workspace with no documents."""

    def test_query_empty_workspace_returns_no_sources(self, rag_pipeline):
        empty_ws = f"ws_empty_{uuid.uuid4().hex[:8]}"
        result = rag_pipeline.query_workspace(
            workspace_id=empty_ws,
            question="What is NLP?",
            routing_path="baseline",
        )
        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) == 0

    def test_empty_workspace_diagnostics_present(self, rag_pipeline):
        empty_ws = f"ws_empty_{uuid.uuid4().hex[:8]}"
        result = rag_pipeline.query_workspace(
            workspace_id=empty_ws,
            question="What is NLP?",
            routing_path="baseline",
        )
        diag = result["diagnostics"]
        assert diag["max_score"] == 0.0
        assert diag["avg_score"] == 0.0
        assert diag["retrieved_count"] == 0


# ============================================================================
# 11. Duplicate file detection
# ============================================================================


class TestDuplicateFileDetection:
    """Re-ingesting the same file must raise DuplicateFileError."""

    def test_duplicate_raises(self, rag_pipeline, ingested_workspace):
        from src.domain.exceptions import DuplicateFileError

        with pytest.raises(DuplicateFileError):
            rag_pipeline.ingest_document(
                workspace_id=ingested_workspace,
                file_path=_get_sample_docx_path(),
                filename="nlp_test.docx",
                openai_api_key=None,
            )


# ============================================================================
# 12. Token estimation utility
# ============================================================================


class TestTokenEstimation:
    """Validates the tiktoken-based token estimation helper."""

    def test_empty_string_returns_zero(self):
        from src.utils.observability import estimate_token_count

        assert estimate_token_count("") == 0

    def test_known_english_string(self):
        from src.utils.observability import estimate_token_count

        count = estimate_token_count("hello world")
        assert isinstance(count, int)
        assert count > 0

    def test_returns_integer(self):
        from src.utils.observability import estimate_token_count

        count = estimate_token_count("Natural Language Processing is a subfield of AI.")
        assert isinstance(count, int)


# Import at module level for class-level attribute access
from src.infrastructure.vectorstores.qdrant_store import QdrantStoreService


# ============================================================================
# 13. Transactional Ingestion Rollback on Failure
# ============================================================================


class TestIngestionRollback:
    """Validates that a failed ingestion cleans up any partially written vector points."""

    def test_ingestion_rollback_on_failure(self, rag_pipeline):
        import uuid
        from unittest.mock import patch
        
        ws_err = f"ws_rollback_{uuid.uuid4().hex[:8]}"
        file_path = _get_sample_docx_path()
        
        original_store_chunks = rag_pipeline.vector_store.store_chunks
        call_count = 0
        
        def mock_store_chunks(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated ingestion crash mid-way")
            return original_store_chunks(*args, **kwargs)
            
        with patch.object(rag_pipeline.vector_store, "store_chunks", side_effect=mock_store_chunks):
            with pytest.raises(RuntimeError) as exc_info:
                rag_pipeline.ingest_document(
                    workspace_id=ws_err,
                    file_path=file_path,
                    filename="nlp_test_failed.docx",
                    openai_api_key=None
                )
            assert "Simulated ingestion crash" in str(exc_info.value)
            
        # Verify that all vectors under the new workspace were rolled back (i.e. collections are empty for this workspace)
        vecs_minilm, labels_minilm = rag_pipeline.vector_store.get_workspace_vectors(
            ws_err, rag_pipeline.vector_store.COLLECTION_MINILM
        )
        vecs_bge, labels_bge = rag_pipeline.vector_store.get_workspace_vectors(
            ws_err, rag_pipeline.vector_store.COLLECTION_BGE
        )
        
        assert len(labels_minilm) == 0
        assert vecs_minilm.shape[0] == 0
        assert len(labels_bge) == 0
        assert vecs_bge.shape[0] == 0

