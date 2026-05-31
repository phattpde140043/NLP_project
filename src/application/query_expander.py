import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.domain.services.query_config import QueryControlPlaneConfig
from src.utils.logger import logger
from src.infrastructure.llm.chat_model import ChatModelService
from src.infrastructure.embeddings.embedding_factory import EmbeddingFactory

class QueryExpander:
    """Stage 5 Control Plane: Expected Utility HyDE expansion & Entity-Weighted Lexical Anchor Guardrails."""

    def __init__(self, config: Optional[QueryControlPlaneConfig] = None):
        self.config = config or QueryControlPlaneConfig()

    def extract_weighted_anchors(self, text: str) -> Dict[str, float]:
        """Extracts technical identifiers, acronyms, and codes with specific architectural weights."""
        anchors: Dict[str, float] = {}
        
        # Regex patterns to capture highly specific technical classes of anchors
        patterns = {
            "stack_trace": r"\b(?:at\s+[a-zA-Z0-9_$]+\.[a-zA-Z0-9_$]+|[a-zA-Z0-9_$]+Exception)\b",
            "api_name": r"\b(?:[A-Z][a-z0-9]+){2,}\b|\b[a-z]+[A-Z][a-zA-Z0-9]*\b", # CamelCase or camelCase
            "error_code": r"\b(?:ERR_[A-Z0-9_]+|[A-Z]{3,}_[A-Z0-9_]+|\b\d{3}\b)\b", # ERR_CONNECTION_RESET, 404, etc.
            "sql_table": r"\b(?:tbl_[a-z0-9_]+|users|documents|workspaces|accounts|logs)\b",
            "config_key": r"\b[a-z0-9_]+\.[a-z0-9_]+(?:\.[a-z0-9_]+)*\b", # settings.storage.path
            "env_var": r"\b[A-Z0-9_]{3,}\b", # DATABASE_URL, PORT
            "file_path": r"\b[a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]{2,4}\b", # config.json, src/main.py
            "k8s_resource": r"\b(?:pod|deployment|service|ingress|configmap|secret|namespace)\b",
            "rfc_spec": r"\b(?:RFC\s*\d+|IEEE\s*\d+|ISO\s*\d+)\b",
            "version_id": r"\bv\d+\.\d+(?:\.\d+)?\b" # v1.2.3
        }
        
        text_lower = text.lower()
        
        # 1. Evaluate specific pattern weights
        for cat, pattern in patterns.items():
            matches = re.findall(pattern, text)
            weight = self.config.anchor_weights.get(cat, 1.0)
            for match in matches:
                # Store match string alongside the highest weight it qualifies for
                cleaned = match.lower().strip()
                if cleaned:
                    anchors[cleaned] = max(anchors.get(cleaned, 0.0), weight)
                    
        # 2. Extract standard fallback alphanumeric tokens as base anchors (W=1.0)
        all_words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text)
        default_weight = self.config.anchor_weights.get("default", 1.0)
        for word in all_words:
            cleaned = word.lower().strip()
            if cleaned and cleaned not in anchors:
                anchors[cleaned] = default_weight
                
        return anchors

    def calculate_anchor_retention(self, query_anchors: Dict[str, float], hyde_text: str) -> Tuple[float, List[str]]:
        """Calculates the Entity-Weighted Lexical Anchor Retention Rate."""
        if not query_anchors:
            return 1.0, []
            
        hyde_lower = hyde_text.lower()
        
        # Identify matching anchors preserved in the hypothetical document
        total_query_weight = sum(query_anchors.values())
        matched_weight = 0.0
        missing_anchors = []
        
        for anchor, weight in query_anchors.items():
            # Check for exact substring match in the generated HyDE text
            if anchor in hyde_lower:
                matched_weight += weight
            else:
                missing_anchors.append(anchor)
                
        retention = matched_weight / total_query_weight if total_query_weight > 0 else 1.0
        return retention, missing_anchors

    def evaluate_hyde_utility(self, query: str, entropy: float, top_gap: float) -> Tuple[bool, float]:
        """Expected Utility Gate: decides if triggering HyDE is economically rational."""
        # 1. Estimate expected recall gain
        # Ambiguous technical queries see higher recall benefits from hypothetical expansion
        length_factor = min(1.0, len(query.split()) / 8.0)
        recall_gain = 0.45 * (entropy / 4.0) + 0.2 * (1.0 - length_factor) + 0.15 * (0.5 - min(0.5, top_gap))
        
        # 2. Model costs & penalties
        latency_penalty = 0.25 # heavy penalty for dynamic API call
        token_cost = 0.05
        
        expected_utility = recall_gain - (latency_penalty + token_cost)
        
        should_trigger = expected_utility > self.config.hyde_expected_utility_threshold
        
        logger.info(f"HyDE Expected Utility: raw_recall_gain={recall_gain:.4f}, expected_utility={expected_utility:.4f}, should_trigger={should_trigger}")
        return should_trigger, expected_utility

    def generate_hyde_hypothesis(self, query: str, openai_api_key: Optional[str] = None) -> str:
        """Generates the hypothetical answer document using LLM Service or local mock fallback."""
        prompt = (
            f"You are an expert technical assistant. Write a short paragraph answering this question. "
            f"Be precise, use technical identifiers, API names, error codes, and paths exactly where relevant. "
            f"Question: {query}"
        )
        
        # Check for valid API key to run actual LLM synthesis
        if openai_api_key and openai_api_key.strip().startswith("sk-"):
            try:
                llm_service = ChatModelService(openai_api_key=openai_api_key.strip())
                hypothesis = llm_service.generate_answer(prompt, "System: Generate hypothetical document only.")
                logger.info("Successfully generated HyDE hypothesis using Cloud LLM.")
                return hypothesis
            except Exception as e:
                logger.error(f"Failed to generate HyDE via Cloud LLM: {str(e)}")
                
        # Graceful Local Fallback: Simulate mock hypothesis that duplicates technical tokens to aid retrieval
        logger.info("Falling back to local high-fidelity technical mock expansion.")
        tokens = query.split()
        # Duplicate key technical words, append dummy technical sentences
        tech_words = [t for t in tokens if len(t) > 3 or t.isupper()]
        tech_context = " ".join(tech_words)
        
        return (
            f"Hypothetical technical document answering the query. "
            f"The active system logs record the following keys and parameters: {tech_context}. "
            f"This relates to an internal class interface processing {tech_context} configurations. "
            f"Error triggers are validated against {tech_context} attributes and return codes."
        )

    def validate_double_lock(
        self, 
        query: str, 
        hyde_text: str, 
        embedding_provider, 
        query_vector: np.ndarray
    ) -> Tuple[bool, float, float]:
        """Double-Lock Guardrail check to reject semantic drift and Terminology Hallucinations."""
        # Lock 1: Semantic Cosine Constraint
        try:
            hyde_vector = embedding_provider.embed_query(hyde_text)
            hyde_arr = np.array(hyde_vector)
            q_arr = np.array(query_vector)
            
            # Simple Cosine similarity via numpy
            cosine_sim = np.dot(q_arr, hyde_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(hyde_arr) + 1e-12)
            cosine_sim = float(cosine_sim)
        except Exception as e:
            logger.error(f"Failed to compute HyDE cosine similarity: {str(e)}")
            cosine_sim = 1.0 # fallback accept
            
        # Lock 2: Entity-Weighted Lexical Anchor Preservation
        q_anchors = self.extract_weighted_anchors(query)
        retention_rate, missing = self.calculate_anchor_retention(q_anchors, hyde_text)
        
        # Verify both constraints are satisfied
        semantic_lock = cosine_sim >= self.config.hyde_cosine_threshold
        lexical_lock = retention_rate >= self.config.hyde_min_weighted_retention
        
        is_accepted = semantic_lock and lexical_lock
        
        logger.info(f"HyDE Double-Lock Guardrails: Cosine={cosine_sim:.4f} (Pass={semantic_lock}), WeightedRetention={retention_rate:.4f} (Pass={lexical_lock}) -> Accepted={is_accepted}")
        if missing:
            logger.debug(f"Missing lexical anchors: {missing}")
            
        return is_accepted, cosine_sim, retention_rate
