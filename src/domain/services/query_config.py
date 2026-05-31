from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class CalibrationProfile:
    """Dynamic and percentile calibration mappings for embedding models."""
    model_name: str
    base_threshold: float = 0.35
    hyde_cosine_threshold: float = 0.78
    high_entropy_cutoff: float = 3.5

@dataclass
class QueryControlPlaneConfig:
    """Configurable hyperparameters for the 8-Stage Adaptive Retrieval Control Plane."""
    # Stage 1: Routing & Calibration
    routing_ema_beta: float = 0.7
    routing_hysteresis_delta: float = 0.1
    
    # Stage 2: Parallel Hedged Search & Path Blending
    parallel_hedged_timeout_ms: float = 40.0
    progressive_ann_alpha: float = 0.7
    
    # Stage 5: Expected Utility & Entity Weighted Guardrails
    hyde_entropy_cutoff: float = 3.5
    hyde_cosine_threshold: float = 0.78
    hyde_min_weighted_retention: float = 0.85
    hyde_expected_utility_threshold: float = 0.0
    
    # Stage 6: Rerank Benefit & Exploration Budget
    rerank_expected_gain_threshold: float = 0.05
    rerank_exploration_rate: float = 0.08  # 8% exploration budget
    
    # Stage 7: 2-Stage Cache & Centroid Drift
    cache_lambda_decay: float = 1e-5  # Decay coefficient
    cache_centroid_drift_theta: float = 0.08  # Invalidation threshold
    
    # Category Weights for Entity-Weighted Lexical Anchors
    anchor_weights: Dict[str, float] = field(default_factory=lambda: {
        "stack_trace": 3.5,
        "api_name": 3.0,
        "class_name": 3.0,
        "function_name": 3.0,
        "error_code": 3.0,
        "sql_table": 2.5,
        "config_key": 2.5,
        "env_var": 2.5,
        "file_path": 2.5,
        "k8s_resource": 2.5,
        "rfc_spec": 2.0,
        "version_id": 2.0,
        "default": 1.0
    })
