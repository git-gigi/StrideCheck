import numpy as np
from src.config import LANDMARKS
from src.gait_segmenter import GaitCycle
from src.models import GaitFeatures

class FeatureExtractor:

    def extract_all(self, cycles: list[GaitCycle], angles: dict[str, np.ndarray], landmarks: np.ndarray, fps: float) -> list[GaitFeatures]:
        features = [self.extract_cycle(c, angles, landmarks, fps) for c in cycles]
        self._fill_symmetry(cycles, features)
        return features


    def extract_cycle(
        self, cycle: GaitCycle,
        angles: dict[str, np.ndarray],
        landmarks: np.ndarray,
        fps: float
        ) -> GaitFeatures:

        suffix = "_r" if cycle.side == "right" else "_l"
        knee_key = f"knee{suffix}"
        hip_key = f"hip{suffix}"
        ankle_key = f"ankle{suffix}"

        knee_values = angles[knee_key][cycle.start_frame : cycle.end_frame + 1]
        hip_values = angles[hip_key][cycle.start_frame : cycle.end_frame + 1]
        trunk_lean_values = angles["trunk_lean"][cycle.start_frame : cycle.end_frame + 1]

        # Vertical Oscillation — peak-to-peak normalized by body height in frame
        hip_lm_idx = LANDMARKS["right_hip"] if cycle.side == "right" else LANDMARKS["left_hip"]
        shoulder_lm_idx = LANDMARKS["right_shoulder"] if cycle.side == "right" else LANDMARKS["left_shoulder"]
        hip_y = landmarks[cycle.start_frame : cycle.end_frame + 1, hip_lm_idx, 1]

        # Overstride — detect at foot strike (ankle lowest point)
        ankle_lm_idx = LANDMARKS["right_ankle"] if cycle.side == "right" else LANDMARKS["left_ankle"]
        ankle_y_series = landmarks[cycle.start_frame:cycle.end_frame + 1, ankle_lm_idx, 1]
        ankle_y_avg = np.mean(ankle_y_series)
        shoulder_y_avg = np.mean(landmarks[cycle.start_frame:cycle.end_frame + 1, shoulder_lm_idx, 1])
        body_height = abs(ankle_y_avg - shoulder_y_avg)
        vert_osc = float(np.max(hip_y) - np.min(hip_y))
        if body_height > 0.01:
            vert_osc /= body_height

        strike_offset = int(np.argmax(ankle_y_series))
        strike_frame = cycle.start_frame + strike_offset

        foot_idx = LANDMARKS["right_foot_index"] if cycle.side == "right" else LANDMARKS["left_foot_index"]
        heel_idx = LANDMARKS["right_heel"] if cycle.side == "right" else LANDMARKS["left_heel"]
        direction = 1.0 if landmarks[strike_frame, foot_idx, 0] > landmarks[strike_frame, heel_idx, 0] else -1.0

        ankle_x = landmarks[strike_frame, ankle_lm_idx, 0]
        hip_x = landmarks[strike_frame, hip_lm_idx, 0]

        return GaitFeatures(
            cadence=(60.0 / cycle.duration_sec) * 2,
            knee_rom=float(max(knee_values) - min(knee_values)),
            knee_peak=180.0 - float(min(knee_values)),
            hip_extension=min(hip_values),
            vertical_oscillation=vert_osc,
            trunk_lean_mean=np.mean(trunk_lean_values),
            trunk_lean_std=np.std(trunk_lean_values),
            stride_symmetry=1.0,
            overstride_index=float((ankle_x - hip_x) * direction),
            key_frame_idx=cycle.start_frame,
        )

    def _fill_symmetry(self, cycles: list[GaitCycle], features: list[GaitFeatures]):
        for i, cycle in enumerate(cycles):
            opposite = "left" if cycle.side == "right" else "right"
            for j in range(i + 1, len(cycles)):
                if cycles[j].side == opposite:
                    features[i].stride_symmetry = cycle.duration_sec / cycles[j].duration_sec
                    break