from typing import Protocol, runtime_checkable
from src.models import GaitFeatures, AnalysisResult


@runtime_checkable
class FormAnalyzer(Protocol):
    """Strategy interface for gait analysis."""
    def analyze(self, features: list[GaitFeatures]) -> AnalysisResult:
        ...
        
        
