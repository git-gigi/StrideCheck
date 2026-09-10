import numpy as np
from src.config import LANDMARKS, JOINT_ANGLES

class AngleCalculator:

    def __init__(self, landmarks: np.ndarray):
        self.landmarks = landmarks
        self.angles = {}

    def compute_all_angles(self) -> dict[str, np.ndarray]:

        for angle_name, (a_name, b_name, c_name) in JOINT_ANGLES.items():
            a_idx = LANDMARKS[a_name]
            b_idx = LANDMARKS[b_name]
            c_idx = LANDMARKS[c_name]
            

            self.angles[angle_name] = []
            for frame_landmarks in self.landmarks:
                self.angles[angle_name].append(compute_angle_3d(
                    frame_landmarks[a_idx, :3], 
                    frame_landmarks[b_idx, :3], 
                    frame_landmarks[c_idx, :3]
                    ))
            
            self.angles[angle_name] = np.array(self.angles[angle_name])

        self.angles["trunk_lean"] = self._compute_trunk_lean_angles(self.landmarks)
        return self.angles

    def _compute_trunk_lean_angles(self, landmarks: np.ndarray) -> np.ndarray:
        l_sh = LANDMARKS["left_shoulder"]
        r_sh = LANDMARKS["right_shoulder"]
        l_hip = LANDMARKS["left_hip"]
        r_hip = LANDMARKS["right_hip"]

        trunk_lean_angles = []
        vertical_vec = np.array([0.0, -1.0])  # Use only 2D (X, Y)

        for frame_lm in landmarks:
            # We still average the shoulders, but we only take X and Y (indices 0 and 1)
            mid_shoulder = (frame_lm[l_sh, :2] + frame_lm[r_sh, :2]) / 2.0
            mid_hip = (frame_lm[l_hip, :2] + frame_lm[r_hip, :2]) / 2.0
            trunk_vec = mid_shoulder - mid_hip

            cos_theta = np.dot(trunk_vec, vertical_vec) / np.linalg.norm(trunk_vec)
            angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            trunk_lean_angles.append(angle_deg)

        return np.array(trunk_lean_angles)



def compute_angle_3d(a, b, c) -> float:
    ab = a - b
    bc = c - b
    
    # Linear Algebra: ab · bc = |ab||bc|cos(theta)
    cos_theta = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))

    # clip to [-1, 1], because arccos is undefined outside this range
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
