from src.models import GaitFeatures, AnalysisResult, FormError, CycleResult
from src.config import THRESHOLDS, LANDMARKS
from .protocol import FormAnalyzer


class HeuristicAnalyzer(FormAnalyzer):

    def analyze(self, features: list[GaitFeatures]) -> AnalysisResult:
        cycle_results = []

        for cycle_id, f in enumerate(features):
            errors = self._evaluate_cycle_errors(f)

            label = "good_form" if len(errors) == 0 else "poor form"
            
            cycle_results.append(
                CycleResult(
                    cycle_id=cycle_id,
                    label=label,
                    confidence=1.0,
                    errors=errors,
                    key_frame_idx=f.key_frame_idx
                )
            )

        clean_cycles = sum(1 for c in cycle_results if len(c.errors) == 0)

        overall_score = clean_cycles / len(features) if features else 1.0
        

        return AnalysisResult(overall_score=overall_score, cycle_results=cycle_results)
    
    def _evaluate_cycle_errors(self, f: GaitFeatures) -> list[FormError]:
        errors = []
        # TODO: verificare warning ed error (per ora solo warning)
        if f.overstride_index > THRESHOLDS["overstride_max"]:
            errors.append(
                FormError(
                    error_type="overstriding",
                    severity="warning",
                    message="Landing ahead of center of gravity",
                    suggestion="Shorten your stride, increase cadence",
                    landmarks_involved=[
                        LANDMARKS["left_ankle"],
                        LANDMARKS["right_ankle"],
                        LANDMARKS["left_hip"],
                        LANDMARKS["right_hip"]
                    ]
                )
            )
        if f.cadence < THRESHOLDS["cadence_min"]:
            errors.append(
                FormError(
                    error_type="low_cadence",
                    severity="warning",
                    message="Cadence below 160 spm",
                    suggestion="Increase cadence to 170-180 spm",
                    landmarks_involved=[
                        LANDMARKS["left_foot_index"],
                        LANDMARKS["right_foot_index"],
                    ]
                )
            )
        if f.vertical_oscillation > THRESHOLDS["vertical_oscillation_max"]:
            errors.append(
                FormError(
                    error_type="excessive_bounce",
                    severity="warning",
                    message="Excessive vertical oscillation",
                    suggestion="Reduce vertical bouncing",
                    landmarks_involved=[
                        LANDMARKS["left_hip"],
                        LANDMARKS["right_hip"]
                    ]
                )
            )
        if f.trunk_lean_mean > THRESHOLDS["trunk_lean_max"]:
            errors.append(
                FormError(
                    error_type="forward_lean",
                    severity="warning",
                    message="Trunk lean above 10°",
                    suggestion="Reduce forward lean",
                    landmarks_involved=[
                        LANDMARKS["left_hip"],
                        LANDMARKS["right_hip"],
                        LANDMARKS["left_shoulder"],
                        LANDMARKS["right_shoulder"],
                    ]
                )
            )
        if abs(f.stride_symmetry - 1) > THRESHOLDS["asymmetry_tolerance"] :
            errors.append(
                FormError(
                    error_type="asymmetry",
                    severity="warning",
                    message="Stride asymmetry above 0.15",
                    suggestion="Improve bilateral symmetry",
                    landmarks_involved=[
                        LANDMARKS["left_knee"],
                        LANDMARKS["right_knee"],
                        LANDMARKS["left_ankle"],
                        LANDMARKS["right_ankle"],
                    ]
                )
            )
        if f.knee_peak < THRESHOLDS["knee_peak_min"]:
            errors.append(
                FormError(
                    error_type="insufficient_knee_drive",
                    severity="warning",
                    message="Knee flextion below 70°",
                    suggestion="Increase flexion to 70-85° range",
                    landmarks_involved=[
                        LANDMARKS["left_knee"],
                        LANDMARKS["right_knee"]
                    ]
                )
            )
        return errors
        
