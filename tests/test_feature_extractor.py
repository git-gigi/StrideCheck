import numpy as np
import pytest
from src.gait_segmenter import GaitCycle
from src.models import GaitFeatures
from src.feature_extractor import FeatureExtractor


@pytest.fixture
def synthetic_cycle_and_data():
    """Builds a realistic synthetic gait cycle with known signals."""
    fps = 30.0
    start_frame = 10
    end_frame = 30  # 20 frames = 2/3 sec -> cadence = 60 / (20/30) = 90 spm (single side) or stride cycle
    duration_sec = (end_frame - start_frame) / fps

    cycle = GaitCycle(
        start_frame=start_frame,
        end_frame=end_frame,
        duration_sec=duration_sec,
        side="right",
    )

    n_total_frames = 50
    # Knee angle: from 60° to 150° (ROM = 90°, peak = 150°)
    knee_r = np.full(n_total_frames, 60.0)
    knee_r[start_frame:end_frame] = np.linspace(60.0, 150.0, end_frame - start_frame)

    knee_l = np.full(n_total_frames, 60.0)
    hip_r = np.linspace(160.0, 200.0, n_total_frames)
    hip_l = np.linspace(160.0, 200.0, n_total_frames)
    ankle_r = np.full(n_total_frames, 90.0)
    ankle_l = np.full(n_total_frames, 90.0)
    trunk_lean = np.full(n_total_frames, 8.0)

    angles = {
        "knee_r": knee_r,
        "knee_l": knee_l,
        "hip_r": hip_r,
        "hip_l": hip_l,
        "ankle_r": ankle_r,
        "ankle_l": ankle_l,
        "trunk_lean": trunk_lean,
    }

    # Landmarks: (n_frames, 33, 4)
    landmarks = np.zeros((n_total_frames, 33, 4))
    # Hip landmark index 24 (right hip) — Y oscillation
    landmarks[:, 24, 1] = 0.5 + 0.05 * np.sin(np.linspace(0, 2 * np.pi, n_total_frames))

    # Body height references: shoulder at Y=0.2, ankle at Y=0.8
    landmarks[:, 12, 1] = 0.2   # right shoulder Y
    landmarks[:, 28, 1] = 0.8   # right ankle Y (default)

    # Simulate foot strike: ankle Y peaks (closest to ground) at frame 15
    landmarks[15, 28, 1] = 0.95  # highest Y = lowest point = foot strike

    # At foot strike (frame 15): ankle ahead of hip -> overstride
    landmarks[15, 28, 0] = 0.6   # right ankle X
    landmarks[15, 24, 0] = 0.55  # right hip X

    # Foot orientation (facing right): foot_index(32) X > heel(30) X
    landmarks[15, 32, 0] = 0.65
    landmarks[15, 30, 0] = 0.55

    return cycle, angles, landmarks, fps


class TestFeatureExtractor:
    def test_extract_returns_gait_features_instance(self, synthetic_cycle_and_data):
        cycle, angles, landmarks, fps = synthetic_cycle_and_data
        extractor = FeatureExtractor()
        features = extractor.extract_cycle(cycle, angles, landmarks, fps)

        assert isinstance(features, GaitFeatures)

    def test_cadence_calculation(self, synthetic_cycle_and_data):
        cycle, angles, landmarks, fps = synthetic_cycle_and_data
        extractor = FeatureExtractor()
        features = extractor.extract_cycle(cycle, angles, landmarks, fps)

        # Cadence in steps per minute: (60 / duration) * 2
        expected_cadence = (60.0 / cycle.duration_sec) * 2
        assert np.isclose(features.cadence, expected_cadence, atol=0.1)

    def test_knee_rom_and_peak_calculation(self, synthetic_cycle_and_data):
        cycle, angles, landmarks, fps = synthetic_cycle_and_data
        extractor = FeatureExtractor()
        features = extractor.extract_cycle(cycle, angles, landmarks, fps)

        # knee_r included angle goes 60->150, so clinical flexion peak = 180 - 60 = 120
        assert np.isclose(features.knee_peak, 120.0, atol=0.5)
        assert np.isclose(features.knee_rom, 90.0, atol=0.5)

    def test_trunk_lean_mean_and_std(self, synthetic_cycle_and_data):
        cycle, angles, landmarks, fps = synthetic_cycle_and_data
        extractor = FeatureExtractor()
        features = extractor.extract_cycle(cycle, angles, landmarks, fps)

        assert np.isclose(features.trunk_lean_mean, 8.0, atol=0.1)
        assert np.isclose(features.trunk_lean_std, 0.0, atol=0.1)

    def test_vertical_oscillation_is_positive(self, synthetic_cycle_and_data):
        cycle, angles, landmarks, fps = synthetic_cycle_and_data
        extractor = FeatureExtractor()
        features = extractor.extract_cycle(cycle, angles, landmarks, fps)

        # Peak-to-peak normalized by body height — should be positive
        assert features.vertical_oscillation > 0.0

    def test_overstride_index_calculation(self, synthetic_cycle_and_data):
        cycle, angles, landmarks, fps = synthetic_cycle_and_data
        extractor = FeatureExtractor()
        features = extractor.extract_cycle(cycle, angles, landmarks, fps)

        # At foot strike (frame 15), ankle_x=0.6, hip_x=0.55, facing right -> overstride = 0.05
        assert np.isclose(features.overstride_index, 0.05, atol=0.01)

    def test_extract_all_cycles(self, synthetic_cycle_and_data):
        cycle, angles, landmarks, fps = synthetic_cycle_and_data
        extractor = FeatureExtractor()
        features_list = extractor.extract_all([cycle, cycle], angles, landmarks, fps)

        assert isinstance(features_list, list)
        assert len(features_list) == 2
        assert all(isinstance(f, GaitFeatures) for f in features_list)
