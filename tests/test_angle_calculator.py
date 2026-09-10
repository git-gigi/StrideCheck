import numpy as np
import pytest
from src.angle_calculator import compute_angle_3d, AngleCalculator
from src.config import JOINT_ANGLES, LANDMARKS


class TestComputeAngle3D:
    def test_orthogonal_angle_90_degrees(self):
        # Vertex at origin, vectors along X and Y axes
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        angle = compute_angle_3d(a, b, c)
        assert np.isclose(angle, 90.0)

    def test_straight_line_180_degrees(self):
        # Opposite directions along X axis
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([-1.0, 0.0, 0.0])
        angle = compute_angle_3d(a, b, c)
        assert np.isclose(angle, 180.0)

    def test_collinear_same_direction_0_degrees(self):
        # Same direction along X axis
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([2.0, 0.0, 0.0])
        angle = compute_angle_3d(a, b, c)
        assert np.isclose(angle, 0.0)

    def test_45_degrees_angle(self):
        # Vertex at origin, vectors along X and diagonal (1, 1, 0)
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 1.0, 0.0])
        angle = compute_angle_3d(a, b, c)
        assert np.isclose(angle, 45.0)

    def test_3d_translated_vertex(self):
        # Test when vertex is not at origin
        offset = np.array([10.0, 20.0, 30.0])
        a = np.array([1.0, 0.0, 0.0]) + offset
        b = np.array([0.0, 0.0, 0.0]) + offset
        c = np.array([0.0, 1.0, 0.0]) + offset
        angle = compute_angle_3d(a, b, c)
        assert np.isclose(angle, 90.0)

    def test_floating_point_clipping_stability(self):
        # Test potential floating point overflow where cos > 1.0 or < -1.0
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])  # Exactly same vector
        angle = compute_angle_3d(a, b, c)
        assert not np.isnan(angle)
        assert np.isclose(angle, 0.0)


class TestAngleCalculator:
    @pytest.fixture
    def synthetic_landmarks(self):
        n_frames = 20
        landmarks = np.zeros((n_frames, 33, 4))
        
        # Give valid coordinates for required landmarks
        for frame in range(n_frames):
            # Left side
            landmarks[frame, LANDMARKS["left_shoulder"]] = [0.0, 1.0, 0.0, 1.0]
            landmarks[frame, LANDMARKS["left_hip"]] = [0.0, 0.5, 0.0, 1.0]
            landmarks[frame, LANDMARKS["left_knee"]] = [0.0, 0.0, 0.0, 1.0]
            landmarks[frame, LANDMARKS["left_ankle"]] = [0.5, 0.0, 0.0, 1.0]
            landmarks[frame, LANDMARKS["left_foot_index"]] = [0.7, 0.0, 0.0, 1.0]
            
            # Right side
            landmarks[frame, LANDMARKS["right_shoulder"]] = [0.0, 1.0, 0.0, 1.0]
            landmarks[frame, LANDMARKS["right_hip"]] = [0.0, 0.5, 0.0, 1.0]
            landmarks[frame, LANDMARKS["right_knee"]] = [0.0, 0.0, 0.0, 1.0]
            landmarks[frame, LANDMARKS["right_ankle"]] = [0.5, 0.0, 0.0, 1.0]
            landmarks[frame, LANDMARKS["right_foot_index"]] = [0.7, 0.0, 0.0, 1.0]

        return landmarks

    def test_compute_all_angles_keys_and_shapes(self, synthetic_landmarks):
        calculator = AngleCalculator(synthetic_landmarks)
        angles = calculator.compute_all_angles()

        assert isinstance(angles, dict)
        assert "trunk_lean" in angles
        assert isinstance(angles["trunk_lean"], np.ndarray)
        assert len(angles["trunk_lean"]) == len(synthetic_landmarks)
        assert np.all(angles["trunk_lean"] >= 0.0)
        assert not np.any(np.isnan(angles["trunk_lean"]))
        for expected_key in JOINT_ANGLES.keys():
            assert expected_key in angles
            assert isinstance(angles[expected_key], np.ndarray)
            assert len(angles[expected_key]) == len(synthetic_landmarks)
            # All angles should be valid degrees in [0, 180]
            assert np.all(angles[expected_key] >= 0.0)
            assert np.all(angles[expected_key] <= 180.0)
            assert not np.any(np.isnan(angles[expected_key]))
