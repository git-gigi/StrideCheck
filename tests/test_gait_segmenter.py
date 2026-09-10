import numpy as np
import pytest
from src.gait_segmenter import GaitSegmenter, GaitCycle
from src.pose_extractor import VideoMetadata


@pytest.fixture
def video_metadata_30fps():
    return VideoMetadata(
        fps=30.0,
        width=1920,
        height=1080,
        total_frames=120,
        duration_sec=4.0
    )


@pytest.fixture
def synthetic_running_angles(video_metadata_30fps):
    """Generates synthetic knee angles simulating 4 seconds of running at 30 FPS.
    
    1.5 strides/sec (90 strides/min per leg = 180 total spm cadence).
    Period = 20 frames per stride.
    Knee angle oscillates between 40 degrees (extension) and 110 degrees (swing flexion).
    """
    n_frames = video_metadata_30fps.total_frames
    t = np.arange(n_frames)
    
    # Left knee: peak at frames 10, 30, 50, 70, 90, 110 (6 peaks)
    knee_l = 75.0 + 35.0 * np.sin(2 * np.pi * t / 20.0 - np.pi / 2)
    
    # Right knee: half-cycle phase shift (peak at frames 20, 40, 60, 80, 100) (5 peaks)
    knee_r = 75.0 + 35.0 * np.sin(2 * np.pi * t / 20.0 + np.pi / 2)
    
    landmarks = np.zeros((n_frames, 33, 4))
    
    angles = {
        "knee_l": knee_l,
        "knee_r": knee_r,
        "hip_l": np.full(n_frames, 160.0),
        "hip_r": np.full(n_frames, 160.0),
        "ankle_l": np.full(n_frames, 90.0),
        "ankle_r": np.full(n_frames, 90.0),
    }
    
    return landmarks, angles


class TestGaitSegmenter:
    def test_segmenter_detects_expected_cycles(self, synthetic_running_angles, video_metadata_30fps):
        landmarks, angles = synthetic_running_angles
        segmenter = GaitSegmenter(landmarks, angles, video_metadata_30fps)
        cycles = segmenter.segment()

        # Left knee has 6 peaks -> 5 cycles
        # Right knee has 5 peaks -> 4 cycles
        # Total = 9 cycles
        assert isinstance(cycles, list)
        assert len(cycles) == 9

        left_cycles = [c for c in cycles if c.side == "left"]
        right_cycles = [c for c in cycles if c.side == "right"]
        assert len(left_cycles) == 5
        assert len(right_cycles) == 4

    def test_gait_cycle_attributes_and_duration(self, synthetic_running_angles, video_metadata_30fps):
        landmarks, angles = synthetic_running_angles
        segmenter = GaitSegmenter(landmarks, angles, video_metadata_30fps)
        cycles = segmenter.segment()

        for cycle in cycles:
            assert isinstance(cycle, GaitCycle)
            assert cycle.start_frame < cycle.end_frame
            assert cycle.side in ("left", "right")
            
            # Expected duration is (end - start) / fps
            expected_duration = (cycle.end_frame - cycle.start_frame) / video_metadata_30fps.fps
            assert np.isclose(cycle.duration_sec, expected_duration)
            
            # Since period is 20 frames at 30 fps, cycle duration should be 20/30 = ~0.667s
            assert np.isclose(cycle.duration_sec, 20.0 / 30.0, atol=0.05)

    def test_cycles_are_chronologically_sorted(self, synthetic_running_angles, video_metadata_30fps):
        landmarks, angles = synthetic_running_angles
        segmenter = GaitSegmenter(landmarks, angles, video_metadata_30fps)
        cycles = segmenter.segment()

        start_frames = [c.start_frame for c in cycles]
        assert start_frames == sorted(start_frames)

    def test_edge_case_flat_signal(self, video_metadata_30fps):
        n_frames = video_metadata_30fps.total_frames
        landmarks = np.zeros((n_frames, 33, 4))
        angles = {
            "knee_l": np.full(n_frames, 90.0),
            "knee_r": np.full(n_frames, 90.0),
        }
        segmenter = GaitSegmenter(landmarks, angles, video_metadata_30fps)
        cycles = segmenter.segment()

        # Flat signal has no peaks -> 0 cycles without crashing
        assert cycles == []

    def test_edge_case_single_peak_returns_empty(self, video_metadata_30fps):
        n_frames = video_metadata_30fps.total_frames
        knee = np.full(n_frames, 50.0)
        knee[50] = 120.0  # Only one single peak
        
        landmarks = np.zeros((n_frames, 33, 4))
        angles = {
            "knee_l": knee,
            "knee_r": knee,
        }
        segmenter = GaitSegmenter(landmarks, angles, video_metadata_30fps)
        cycles = segmenter.segment()

        # Single peak cannot form a start->end pair
        assert cycles == []

    def test_edge_case_very_short_signal(self):
        short_metadata = VideoMetadata(fps=30.0, width=1920, height=1080, total_frames=5, duration_sec=0.16)
        landmarks = np.zeros((5, 33, 4))
        angles = {
            "knee_l": np.array([50.0, 52.0, 51.0, 50.0, 50.0]),
            "knee_r": np.array([50.0, 52.0, 51.0, 50.0, 50.0]),
        }
        segmenter = GaitSegmenter(landmarks, angles, short_metadata)
        cycles = segmenter.segment()
        assert cycles == []
