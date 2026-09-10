import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Savitzky-Golay Filter
# Solves jitter problems to return reliable angles trough frames

WINDOW_SIZE = 13
POLYORDER = 3

class SavitzkyGolayFilter:
    def __init__(self, window_size: int = WINDOW_SIZE, polyorder: int = POLYORDER):
        self.window_size = window_size
        self.polyorder = polyorder

    def _interpolate_low_confidence(self, landmarks: np.ndarray) -> np.ndarray:
        cleaned = landmarks.copy()

        mask = cleaned[:, :, 3] < 0.5
        cleaned[mask, :3] = np.nan

        n_frames = cleaned.shape[0]

        return pd.DataFrame(cleaned.reshape(n_frames, -1)).interpolate(limit_direction="both").fillna(0).to_numpy().reshape(cleaned.shape)

    def apply_filter(self, landmarks: np.ndarray) -> np.ndarray:
        cleaned = self._interpolate_low_confidence(landmarks)

        filtered = savgol_filter(cleaned, window_length=self.window_size, polyorder=self.polyorder, axis=0)

        return filtered