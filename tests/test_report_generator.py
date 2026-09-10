import pytest
from src.models import AnalysisResult, CycleResult, FormError
from src.report_generator import ReportGenerator, ReportSummary


class TestReportGenerator:
    def test_generate_summary_for_clean_analysis(self):
        generator = ReportGenerator()
        cycle = CycleResult(
            cycle_id=0,
            label="good_form",
            confidence=0.95,
            errors=[],
            key_frame_idx=10,
        )
        analysis_result = AnalysisResult(overall_score=1.0, cycle_results=[cycle, cycle])
        summary = generator.generate_summary(analysis_result)

        assert isinstance(summary, ReportSummary)
        assert summary.overall_score == 1.0
        assert summary.total_cycles == 2
        assert summary.clean_cycles == 2
        assert len(summary.error_counts) == 0
        assert len(summary.recommendations) == 0

    def test_generate_summary_aggregates_errors_and_recommendations(self):
        generator = ReportGenerator()
        err1 = FormError(
            error_type="overstriding",
            severity="warning",
            message="Foot landed too far in front",
            suggestion="Focus on landing under your hips",
            landmarks_involved=[27, 28],
        )
        err2 = FormError(
            error_type="low_cadence",
            severity="warning",
            message="Cadence is below 160 spm",
            suggestion="Aim for 170-180 steps per minute",
            landmarks_involved=[],
        )
        cycle1 = CycleResult(
            cycle_id=0,
            label="overstriding",
            confidence=0.9,
            errors=[err1],
            key_frame_idx=5,
        )
        cycle2 = CycleResult(
            cycle_id=1,
            label="low_cadence",
            confidence=0.9,
            errors=[err2],
            key_frame_idx=25,
        )
        analysis_result = AnalysisResult(overall_score=0.0, cycle_results=[cycle1, cycle2])
        summary = generator.generate_summary(analysis_result)

        assert summary.total_cycles == 2
        assert summary.clean_cycles == 0
        assert summary.error_counts.get("overstriding") == 1
        assert summary.error_counts.get("low_cadence") == 1
        assert len(summary.recommendations) == 2
        assert "Focus on landing under your hips" in summary.recommendations
        assert "Aim for 170-180 steps per minute" in summary.recommendations
