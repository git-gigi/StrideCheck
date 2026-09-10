import pytest
from src.models import GaitFeatures, FormError, CycleResult, AnalysisResult
from src.analyzer.protocol import FormAnalyzer
from src.analyzer.heuristic import HeuristicAnalyzer
from src.analyzer import create_analyzer


@pytest.fixture
def optimal_gait_features() -> GaitFeatures:
    """Fixture returning a cycle with optimal biomechanical features."""
    return GaitFeatures(
        cadence=180.0,
        knee_rom=90.0,
        knee_peak=90.0,
        hip_extension=180.0,
        vertical_oscillation=0.04,
        trunk_lean_mean=6.0,
        trunk_lean_std=1.2,
        stride_symmetry=1.0,
        overstride_index=0.0,
    )


class TestHeuristicAnalyzer:
    def test_satisfies_form_analyzer_protocol(self):
        analyzer = HeuristicAnalyzer()
        assert isinstance(analyzer, FormAnalyzer)

    def test_analyze_good_form_has_no_errors(self, optimal_gait_features):
        analyzer = HeuristicAnalyzer()
        result = analyzer.analyze([optimal_gait_features])

        assert isinstance(result, AnalysisResult)
        assert result.overall_score == 1.0
        assert len(result.cycle_results) == 1
        
        cycle_res = result.cycle_results[0]
        assert isinstance(cycle_res, CycleResult)
        assert cycle_res.label == "good_form"
        assert len(cycle_res.errors) == 0

    def test_detects_overstriding(self, optimal_gait_features):
        optimal_gait_features.overstride_index = 0.15  # Positive value indicates foot strike ahead of COM
        analyzer = HeuristicAnalyzer()
        result = analyzer.analyze([optimal_gait_features])

        errors = result.cycle_results[0].errors
        error_types = [e.error_type for e in errors]
        assert "overstriding" in error_types

        overstride_err = next(e for e in errors if e.error_type == "overstriding")
        assert isinstance(overstride_err, FormError)
        assert overstride_err.severity in ("warning", "error")
        assert len(overstride_err.suggestion) > 0
        assert len(overstride_err.landmarks_involved) > 0

    def test_detects_low_cadence(self, optimal_gait_features):
        optimal_gait_features.cadence = 150.0  # < 160 spm threshold
        analyzer = HeuristicAnalyzer()
        result = analyzer.analyze([optimal_gait_features])

        errors = result.cycle_results[0].errors
        error_types = [e.error_type for e in errors]
        assert "low_cadence" in error_types

    def test_detects_excessive_bounce(self, optimal_gait_features):
        optimal_gait_features.vertical_oscillation = 0.12  # Excessive vertical displacement
        analyzer = HeuristicAnalyzer()
        result = analyzer.analyze([optimal_gait_features])

        errors = result.cycle_results[0].errors
        error_types = [e.error_type for e in errors]
        assert "excessive_bounce" in error_types

    def test_detects_forward_lean(self, optimal_gait_features):
        optimal_gait_features.trunk_lean_mean = 16.0  # > 12° threshold
        analyzer = HeuristicAnalyzer()
        result = analyzer.analyze([optimal_gait_features])

        errors = result.cycle_results[0].errors
        error_types = [e.error_type for e in errors]
        assert "forward_lean" in error_types

    def test_detects_asymmetry(self, optimal_gait_features):
        optimal_gait_features.stride_symmetry = 1.25  # > 0.15 deviation from 1.0
        analyzer = HeuristicAnalyzer()
        result = analyzer.analyze([optimal_gait_features])

        errors = result.cycle_results[0].errors
        error_types = [e.error_type for e in errors]
        assert "asymmetry" in error_types

    def test_detects_insufficient_knee_drive(self, optimal_gait_features):
        optimal_gait_features.knee_peak = 60.0  # < 70° threshold
        analyzer = HeuristicAnalyzer()
        result = analyzer.analyze([optimal_gait_features])

        errors = result.cycle_results[0].errors
        error_types = [e.error_type for e in errors]
        assert "insufficient_knee_drive" in error_types

    def test_overall_score_reflects_error_ratio(self, optimal_gait_features):
        good_cycle = optimal_gait_features
        bad_cycle = GaitFeatures(
            cadence=140.0,
            knee_rom=50.0,
            knee_peak=55.0,
            hip_extension=150.0,
            vertical_oscillation=0.15,
            trunk_lean_mean=18.0,
            trunk_lean_std=5.0,
            stride_symmetry=1.3,
            overstride_index=0.2,
        )
        analyzer = HeuristicAnalyzer()
        result = analyzer.analyze([good_cycle, bad_cycle])

        assert len(result.cycle_results) == 2
        # Score is proportional to clean vs defective cycles (e.g. 0.5)
        assert 0.0 <= result.overall_score < 1.0


class TestAnalyzerFactory:
    def test_create_heuristic_analyzer(self):
        analyzer = create_analyzer("heuristic")
        assert isinstance(analyzer, HeuristicAnalyzer)
        assert isinstance(analyzer, FormAnalyzer)

    def test_create_unknown_analyzer_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown analyzer"):
            create_analyzer("non_existent_method")
