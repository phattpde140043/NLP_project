import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.domain.services.query_config import QueryControlPlaneConfig, CalibrationProfile
from src.utils.logger import logger

class QueryAnalyzer:
    """Stage 1 & 3 Decision Plane: Query Profiling, Calibrated Routing, and Multi-Signal Uncertainty."""
    
    def __init__(self, config: Optional[QueryControlPlaneConfig] = None):
        self.config = config or QueryControlPlaneConfig()
        # Session route smoothing memory (session_id -> vector[exact, semantic, hybrid])
        self.session_ema: Dict[str, np.ndarray] = {}
        
        # Logistic Regression Weight Matrix for Cheap Query Profiler
        # Class order: 0: exact (BM25), 1: semantic (Dense Vector), 2: hybrid
        # Features: [has_regex, symbol_density, caps_ratio, token_rarity, query_length, lexical_dominance]
        self.router_weights = np.array([
            [4.0,  5.0,  3.0,  1.0, -0.5,  3.0],  # exact path favours regex, symbols, caps, Technical Lexical
            [-3.0, -4.0, -3.0, -1.0,  1.0, -2.5],  # semantic path favours natural language, longer queries
            [1.0,  1.5,  1.0,  0.5,  0.2,  1.2]   # hybrid path is balanced
        ])
        self.router_bias = np.array([-1.0, 1.5, 0.5])

    def extract_features(self, query: str) -> np.ndarray:
        """Extracts low-latency lightweight structural features from query string."""
        length = len(query)
        if length == 0:
            return np.zeros(6)
            
        # 1. Regex Match (presence of file paths, exact error codes, identifiers, IP addresses)
        regex_patterns = [
            r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_]+\b",  # file names, dot accessors
            r"\bERR_[A-Z0-9_]+\b",                           # error codes
            r"0x[0-9a-fA-F]+",                              # hex identifiers
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",      # IP addresses
            r"[:/=+\-*{}\[\]]"                              # code operators
        ]
        has_regex = 1.0 if any(re.search(pat, query) for pat in regex_patterns) else 0.0
        
        # 2. Symbol Density
        symbols = re.findall(r"[:/=+\-*{}\[\]_.\(\)#<>&\\/\'\"]", query)
        symbol_density = len(symbols) / length
        
        # 3. Capitalization Ratio
        letters = re.findall(r"[A-Za-z]", query)
        caps = re.findall(r"[A-Z]", query)
        caps_ratio = len(caps) / len(letters) if letters else 0.0
        
        # 4. Token Rarity & Lexical Dominance (highly specific technical keywords)
        tokens = [tok.strip("(),.[]{}'\"") for tok in query.lower().split()]
        rare_keywords = ["api", "error", "exception", "failed", "config", "version", "sql", "db", "table", "port", "kubernetes", "k8s", "docker", "endpoint", "method", "class", "function", "json", "yaml", "xml", "null", "undefined"]
        rare_count = sum(1.0 for tok in tokens if tok in rare_keywords)
        token_rarity = rare_count / max(1.0, len(tokens))
        
        # 5. Query Length factor
        query_length = min(1.0, len(tokens) / 25.0)
        
        # 6. Lexical Dominance (ratio of technical terms to total tokens)
        technical_matches = re.findall(r"\b[A-Z_]{3,}\b|\b[A-Z]?[a-z]+[A-Z][a-zA-Z]*\b|\b[a-zA-Z]+_[a-zA-Z_]+\b", query)
        lexical_dominance = len(technical_matches) / max(1.0, len(tokens))
        
        return np.array([has_regex, symbol_density, caps_ratio, token_rarity, query_length, lexical_dominance])

    def route_query(self, query: str, session_id: Optional[str] = None, active_scores: Optional[List[float]] = None) -> Dict[str, Any]:
        """Stage 1: Cheap Profiling + Percentile-based Dynamic Temperature Calibration + Temporal Routing Hysteresis."""
        features = self.extract_features(query)
        
        # Compute raw classifier routing affinity scores
        logits = np.dot(self.router_weights, features) + self.router_bias
        
        # Percentile-Based Online Calibration (Self-Tuning Temperature T)
        if active_scores and len(active_scores) >= 3:
            s_sorted = np.sort(active_scores)
            p90 = np.percentile(s_sorted, 90)
            p10 = np.percentile(s_sorted, 10)
            t_calibrated = max(0.05, float(p90 - p10) / 2.0)
        else:
            # Fallback default temperature
            t_calibrated = 0.2
            
        # Calibrated Probability Scaling
        exp_logits = np.exp(logits / t_calibrated)
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Class probabilities mapping: [exact, semantic, hybrid]
        raw_prob_dict = {
            "exact": float(probabilities[0]),
            "semantic": float(probabilities[1]),
            "hybrid": float(probabilities[2])
        }
        
        # Temporal Routing Hysteresis & EMA Smoothing (Chống dao động định tuyến)
        if session_id:
            if session_id not in self.session_ema:
                self.session_ema[session_id] = probabilities
            else:
                beta = self.config.routing_ema_beta
                self.session_ema[session_id] = beta * self.session_ema[session_id] + (1.0 - beta) * probabilities
            
            smoothed_probs = self.session_ema[session_id]
        else:
            smoothed_probs = probabilities
            
        # Determine Routing Decision with Hysteresis
        # If the highest score is close to semantic, or difference is within delta, prefer stable route
        argmax_idx = int(np.argmax(smoothed_probs))
        paths = ["exact", "semantic", "hybrid"]
        routing_path = paths[argmax_idx]
        
        return {
            "routing_path": routing_path,
            "raw_affinities": raw_prob_dict,
            "calibrated_temperature": t_calibrated,
            "smoothed_affinities": {paths[i]: float(smoothed_probs[i]) for i in range(3)},
            "features": features.tolist()
        }

    def compute_robust_scores(self, scores: List[float]) -> np.ndarray:
        """Stage 3: Robust MAD Normalization and Clipping to avoid Softmax Saturation."""
        if not scores:
            return np.array([])
            
        s_arr = np.array(scores)
        median = np.median(s_arr)
        mad = np.median(np.abs(s_arr - median))
        
        # Safeguard epsilon division
        epsilon = 1e-6
        normalized = (s_arr - median) / (mad + epsilon)
        
        # Hard clip bound limit to prevent Softmax Saturation [-4, 4]
        return np.clip(normalized, -4.0, 4.0)

    def calculate_entropy(self, robust_scores: np.ndarray, tau: float = 0.25) -> float:
        """Calculates Softmax Entropy over robust normalized scores."""
        if len(robust_scores) == 0:
            return 0.0
            
        # Calibrated softmax calculation
        exp_s = np.exp(robust_scores / tau)
        probs = exp_s / np.sum(exp_s)
        
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        return float(entropy)

    def calculate_jaccard_stability(self, runs: List[List[str]]) -> float:
        """Measures retrieval run overlap stability via Jaccard index."""
        if len(runs) < 2:
            return 1.0
            
        sets = [set(r) for r in runs]
        intersection = set.intersection(*sets)
        union = set.union(*sets)
        
        if not union:
            return 1.0
            
        return float(len(intersection) / len(union))

    def diagnose_uncertainty(self, raw_scores: List[float], sparse_scores: Optional[List[float]] = None, stability_runs: Optional[List[List[str]]] = None) -> Dict[str, Any]:
        """Stage 3 Uncertainty Diagnostics: Multi-Signal Uncertainty Vector U_q calculation."""
        if not raw_scores:
            return {
                "uncertainty_vector": [0.0, 0.0, 0.0, 0.0, 1.0],
                "entropy": 0.0,
                "top_gap": 0.0,
                "variance": 0.0,
                "is_unstable": True,
                "is_uncertain": True
            }
            
        robust_s = self.compute_robust_scores(raw_scores)
        
        # 1. Softmax Entropy H_q
        h_q = self.calculate_entropy(robust_s)
        
        # 2. Ranking frontier top-gap Gap_1,2 (highly robust signal)
        if len(robust_s) >= 2:
            gap_12 = float(robust_s[0] - robust_s[1])
        else:
            gap_12 = 0.0
            
        # 3. Score variance
        variance = float(np.var(robust_s))
        
        # 4. Sparse-Dense Disagreement
        # Measures overlap disagreement between vector and keyword scores
        if sparse_scores and len(sparse_scores) == len(raw_scores):
            agreement = float(np.corrcoef(raw_scores, sparse_scores)[0, 1])
            if np.isnan(agreement):
                agreement = 1.0
            sparse_dense_disagreement = 1.0 - max(0.0, agreement)
        else:
            sparse_dense_disagreement = 0.0
            
        # 5. Jaccard Stability Index
        stability_index = self.calculate_jaccard_stability(stability_runs) if stability_runs else 1.0
        retrieval_instability = 1.0 - stability_index
        
        # Multi-Signal Uncertainty Vector U_q
        u_q = [h_q, gap_12, variance, sparse_dense_disagreement, retrieval_instability]
        
        # High uncertainty heuristic: high entropy or very tiny gap
        is_uncertain = h_q >= self.config.hyde_entropy_cutoff or gap_12 < 0.15
        
        return {
            "uncertainty_vector": u_q,
            "entropy": h_q,
            "top_gap": gap_12,
            "variance": variance,
            "sparse_dense_disagreement": sparse_dense_disagreement,
            "retrieval_instability": retrieval_instability,
            "is_uncertain": is_uncertain
        }

    def measure_graph_aging(self, write_count: int, delete_count: int, traversal_lengths: Optional[List[int]] = None) -> Dict[str, Any]:
        """Research Stage: Online HNSW Aging Metric and Traversal Path Navigation Entropy."""
        total_ops = write_count + delete_count
        
        # 1. Traversal Navigation Entropy (diversity of search paths)
        # Traversal path lengths distribution entropy
        if traversal_lengths and len(traversal_lengths) >= 3:
            lengths, counts = np.unique(traversal_lengths, return_counts=True)
            probs = counts / np.sum(counts)
            navigation_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
            traversal_variance = float(np.var(traversal_lengths))
        else:
            navigation_entropy = 2.0  # nominal high diversity
            traversal_variance = 1.0
            
        # 2. Aging Estimation
        # Degradation escalates with deletion fragmentation and traversal variance collapse
        recall_drop = min(0.3, total_ops * 0.001)  # simulated degradation rate
        graph_imbalance = min(0.4, delete_count * 0.003)
        insert_depth_variance = min(0.3, max(0.0, 0.3 - traversal_variance * 0.05))
        
        aging_score = 0.5 * recall_drop + 0.3 * graph_imbalance + 0.2 * insert_depth_variance
        
        # Navigation degeneracy check (low entropy, traversal path is collapsing)
        is_collapsed = navigation_entropy < 0.8
        
        return {
            "aging_score": float(aging_score),
            "navigation_entropy": navigation_entropy,
            "traversal_variance": traversal_variance,
            "is_collapsed": is_collapsed,
            "trigger_reindex": aging_score > 0.75 or is_collapsed
        }
