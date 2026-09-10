"""Constants and landmark mappings for StrideCheck."""

# MediaPipe Pose landmark indices (only the ones relevant for running gait)
# Full reference: mp.solutions.pose.PoseLandmark

LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# Joint angle definitions: each entry maps an angle name to
# the three landmarks (A, B, C) needed to compute the angle at vertex B.
JOINT_ANGLES = {
    "knee_l": ("left_hip", "left_knee", "left_ankle"),
    "knee_r": ("right_hip", "right_knee", "right_ankle"),
    "hip_l": ("left_shoulder", "left_hip", "left_knee"),
    "hip_r": ("right_shoulder", "right_hip", "right_knee"),
    "ankle_l": ("left_knee", "left_ankle", "left_foot_index"),
    "ankle_r": ("right_knee", "right_ankle", "right_foot_index"),
}

# Biomechanical thresholds based on sports science & literature
THRESHOLDS = {
    "cadence_min": 160.0,              # Minimum optimal cadence in steps/min (spm)
    "knee_peak_min": 70.0,             # Minimum peak knee flexion for proper knee drive (deg)
    "trunk_lean_max": 10.0,            # Maximum acceptable forward trunk lean (deg)
    "asymmetry_tolerance": 0.15,       # Maximum deviation from 1.0 for stride symmetry
    "overstride_max": 0.05,             # Tolerance for normal foot placement ahead of hip (normalized coords)
    "vertical_oscillation_max": 0.08,  # Max peak-to-peak hip oscillation as fraction of body height
}

