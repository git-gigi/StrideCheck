from dataclasses import dataclass
from src.models import AnalysisResult

@dataclass
class ReportSummary:
    overall_score: float
    total_cycles: int
    clean_cycles: int
    error_counts: dict[str, int]
    recommendations: list[str]


class ReportGenerator:
    def generate_summary(self, analysis_result: AnalysisResult) -> ReportSummary:
        error_counts = {}
        recommendations = set()

        for cycle in analysis_result.cycle_results:
            for err in cycle.errors:
                error_counts[err.error_type] = error_counts.get(err.error_type, 0) + 1

                recommendations.add(err.suggestion)
        
        return ReportSummary(
            overall_score=analysis_result.overall_score,
            total_cycles=len(analysis_result.cycle_results),
            clean_cycles=sum(1 for cycle in analysis_result.cycle_results if len(cycle.errors) == 0),
            error_counts=error_counts,
            recommendations=list(recommendations)
        )
        
