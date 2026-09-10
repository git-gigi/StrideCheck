from dataclasses import dataclass


@dataclass
class GaitFeatures:
    cadence: float
    knee_rom: float
    knee_peak: float
    hip_extension: float
    vertical_oscillation: float
    trunk_lean_mean: float
    trunk_lean_std: float
    stride_symmetry: float
    overstride_index: float
    key_frame_idx: int = 0


@dataclass
class FormError:
    error_type: str
    severity: str
    message: str
    suggestion: str
    landmarks_involved: list[int]

@dataclass
class CycleResult:
    cycle_id: int
    label: str
    confidence: float
    errors: list[FormError]
    key_frame_idx: int

@dataclass
class AnalysisResult:
    overall_score: float
    cycle_results: list[CycleResult]


