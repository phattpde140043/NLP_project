import time
import tiktoken
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class PerformanceProfile:
    """Dataclass capturing time profiles (latency spans) for the execution pipeline."""
    spans: Dict[str, float] = field(default_factory=dict)
    
    def start_span(self, name: str) -> float:
        """Starts a named execution span, returning current timestamp."""
        t_start = time.perf_counter()
        self.spans[f"{name}_start"] = t_start
        return t_start
        
    def stop_span(self, name: str) -> float:
        """Stops a named execution span, computing and saving duration in milliseconds."""
        t_stop = time.perf_counter()
        t_start = self.spans.get(f"{name}_start")
        
        if t_start is None:
            return 0.0
            
        duration_ms = (t_stop - t_start) * 1000.0
        self.spans[name] = round(duration_ms, 2)
        
        # Clean up start marker
        self.spans.pop(f"{name}_start", None)
        return round(duration_ms, 2)
        
    def get_metrics_dict(self) -> Dict[str, float]:
        """Returns the dictionary of calculated duration spans (ms), filtering out raw benchmarks."""
        return {k: v for k, v in self.spans.items() if not k.endswith("_start")}


def estimate_token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    """Estimates the exact token count of a given string using specified tiktoken encoder.
    
    Defaults to cl100k_base (standard OpenAI vocabulary).
    """
    if not text:
        return 0
        
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text, disallowed_special=()))
    except Exception:
        # Fallback approximation: 1 token is roughly 4 characters in English, ~2.5 in Vietnamese
        return len(text) // 3
