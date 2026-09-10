import numpy as np
import pytest
from src.models import FormError, CycleResult
from src.video_annotator import VideoAnnotator


class TestVideoAnnotator:
    def test_annotate_frame_preserves_shape_and_type(self):
        annotator = VideoAnnotator()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_landmarks = np.zeros((33, 4))
        # Left knee (25), Left ankle (27)
        highlight_landmarks = [25, 27]

        annotated = annotator.annotate_frame(
            frame=dummy_frame,
            landmarks=dummy_landmarks,
            highlight_landmarks=highlight_landmarks,
        )

        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == (480, 640, 3)
        assert annotated.dtype == np.uint8

    def test_extract_key_frames_returns_annotated_tuples(self):
        annotator = VideoAnnotator()
        error = FormError(
            error_type="overstriding",
            severity="warning",
            message="Foot landed ahead of COM",
            suggestion="Increase cadence",
            landmarks_involved=[27, 28],
        )
        cycle_result = CycleResult(
            cycle_id=0,
            label="overstriding",
            confidence=0.9,
            errors=[error],
            key_frame_idx=15,
        )
        landmarks = np.zeros((30, 33, 4))

        key_frames = annotator.extract_key_frames(
            video_path="dummy_path.mp4",
            landmarks=landmarks,
            cycle_results=[cycle_result],
        )

        assert isinstance(key_frames, list)
        assert len(key_frames) == 1
        frame_idx, frame_img, err = key_frames[0]
        assert frame_idx == 15
        assert isinstance(frame_img, np.ndarray)
        assert err == error
