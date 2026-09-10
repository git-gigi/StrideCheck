from .protocol import FormAnalyzer
from .heuristic import HeuristicAnalyzer

def create_analyzer(method: str = "heuristic") -> FormAnalyzer:
    if method == "heuristic":
        return HeuristicAnalyzer()
    raise ValueError(f"Unknown analyzer: {method}")
