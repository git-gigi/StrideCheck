from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from src.models import AnalysisResult, CycleResult, GaitFeatures
from src.analyzer.protocol import FormAnalyzer
from src.pose_extractor import VideoMetadata
from src.gait_segmenter import GaitCycle
from src.pipeline import AnalysisPipeline


class DummyAnalyzer:
    """A dummy analyzer satisfying the FormAnalyzer protocol for testing."""
    def analyze(self, features: list[GaitFeatures]) -> AnalysisResult:
        cycle_res = [
            CycleResult(
                cycle_id=i,
                label="good_form",
                confidence=0.95,
                errors=[],
                key_frame_idx=10 * i,
            )
            for i in range(len(features))
        ]
        return AnalysisResult(overall_score=1.0, cycle_results=cycle_res)


class TestAnalysisPipeline:
    def test_pipeline_initialization_with_dependency_injection(self):
        analyzer = DummyAnalyzer()
        assert isinstance(analyzer, FormAnalyzer)
        pipeline = AnalysisPipeline(analyzer=analyzer)
        assert pipeline.analyzer is analyzer

    @patch("src.pipeline.PoseExtractor")
    @patch("src.pipeline.GaitSegmenter")
    @patch("src.pipeline.FeatureExtractor")
    def test_pipeline_run_orchestrates_steps(
        self, mock_feature_extractor_cls, mock_gait_segmenter_cls, mock_pose_extractor_cls
    ):
        # Setup mocks
        mock_metadata = VideoMetadata(fps=30.0, width=1920, height=1080, total_frames=60, duration_sec=2.0)
        mock_landmarks = np.zeros((60, 33, 4))
        mock_pose_inst = mock_pose_extractor_cls.return_value
        mock_pose_inst.extract.return_value = (mock_landmarks, mock_metadata)

        mock_segmenter_inst = mock_gait_segmenter_cls.return_value
        mock_cycle = GaitCycle(start_frame=0, end_frame=30, duration_sec=1.0, side="right")
        mock_segmenter_inst.segment.return_value = [mock_cycle]

        mock_feat_inst = mock_feature_extractor_cls.return_value
        mock_features = GaitFeatures(
            cadence=180.0,
            knee_rom=90.0,
            knee_peak=90.0,
            hip_extension=180.0,
            vertical_oscillation=0.05,
            trunk_lean_mean=6.0,
            trunk_lean_std=1.0,
            stride_symmetry=1.0,
            overstride_index=0.0,
        )
        mock_feat_inst.extract_all.return_value = [mock_features]

        analyzer = DummyAnalyzer()
        pipeline = AnalysisPipeline(analyzer=analyzer)
        result, cleaned_landmarks, returned_metadata, returned_angles, returned_cycles, returned_features = pipeline.run("dummy_video.mp4")

        assert isinstance(result, AnalysisResult)
        assert result.overall_score == 1.0
        assert len(result.cycle_results) == 1
        assert result.cycle_results[0].label == "good_form"

        # Verify calls
        mock_pose_inst.extract.assert_called_once_with("dummy_video.mp4")
        mock_feat_inst.extract_all.assert_called_once()
