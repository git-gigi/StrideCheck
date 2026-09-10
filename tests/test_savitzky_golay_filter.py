import numpy as np
import pytest
from src.savitzky_golay_filter import SavitzkyGolayFilter


class TestSavitzkyGolayFilter:
    @pytest.fixture
    def filter_instance(self):
        return SavitzkyGolayFilter(window_size=11, polyorder=3)

    def test_output_shape_matches_input(self, filter_instance):
        n_frames = 50
        landmarks = np.random.rand(n_frames, 33, 4)
        landmarks[:, :, 3] = 1.0  # full confidence
        
        filtered = filter_instance.apply_filter(landmarks)
        assert filtered.shape == landmarks.shape

    def test_low_confidence_interpolation(self, filter_instance):
        n_frames = 30
        landmarks = np.ones((n_frames, 33, 4))
        
        # Introduce low visibility frame in the middle
        landmarks[10:15, :, 3] = 0.1  # Low confidence (< 0.5)
        
        filtered = filter_instance.apply_filter(landmarks)
        
        # Verify no NaN exists in filtered landmarks
        assert not np.any(np.isnan(filtered))

    def test_noise_reduction(self, filter_instance):
        # Create a smooth sinusoidal trajectory
        n_frames = 100
        t = np.linspace(0, 4 * np.pi, n_frames)
        clean_signal = np.sin(t)
        
        # Add gaussian noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.2, size=n_frames)
        noisy_signal = clean_signal + noise
        
        # Format into landmarks tensor (n_frames, 33, 4)
        landmarks = np.zeros((n_frames, 33, 4))
        landmarks[:, 0, 0] = noisy_signal
        landmarks[:, :, 3] = 1.0  # high confidence
        
        filtered = filter_instance.apply_filter(landmarks)
        filtered_signal = filtered[:, 0, 0]
        
        # Mean Squared Error comparison
        noisy_mse = np.mean((noisy_signal - clean_signal) ** 2)
        filtered_mse = np.mean((filtered_signal - clean_signal) ** 2)
        
        # The filter must reduce the error compared to the noisy raw signal
        assert filtered_mse < noisy_mse
