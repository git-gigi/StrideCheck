import numpy as np
import cv2
from cv2.typing import Point
from src.models import FormError
from src.models import CycleResult
from src.gait_segmenter import GaitCycle


POSE_CONNECTIONS = [
    (11, 12),  # Shoulders
    (11, 23),  # Left shoulder -> Left hip
    (12, 24),  # Right shoulder -> Right hip
    (23, 24),  # Hips
    (23, 25),  # Left hip -> Left knee
    (24, 26),  # Right hip -> Right knee
    (25, 27),  # Left knee -> Left ankle
    (26, 28),  # Right knee -> Right ankle
    (27, 29),  # Left ankle -> Left heel
    (28, 30),  # Right ankle -> Right heel
    (29, 31),  # Left heel -> Left foot index
    (30, 32),  # Right heel -> Right foot index
]


class VideoAnnotator:
    
    def extract_key_frames(self, video_path: str, landmarks: np.ndarray, cycle_results: list[CycleResult]) -> list[tuple[int, np.ndarray, FormError]]:
        annotated_frames = []

        for cycle in cycle_results:
            if len(cycle.errors) != 0:
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, cycle.key_frame_idx)
                ret, frame = cap.read()
                cap.release()

                if not ret or frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

                for err in cycle.errors:
                    annotated = self.annotate_frame(frame=frame.copy(), landmarks=landmarks[cycle.key_frame_idx], highlight_landmarks=err.landmarks_involved)
                    annotated_frames.append((cycle.key_frame_idx, annotated, err))

        return annotated_frames

    def generate_annotated_video(self, video_path: str, output_path: str,
                                  landmarks: np.ndarray, cycles: list[GaitCycle],
                                  cycle_results: list[CycleResult]):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Build frame -> error landmarks mapping
        frame_highlights: dict[int, set[int]] = {}
        for cycle, cr in zip(cycles, cycle_results):
            if cr.errors:
                lms = set()
                for err in cr.errors:
                    lms.update(err.landmarks_involved)
                for f in range(cycle.start_frame, cycle.end_frame + 1):
                    frame_highlights.setdefault(f, set()).update(lms)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx < len(landmarks):
                highlight = list(frame_highlights.get(frame_idx, []))
                frame = self.annotate_frame(frame, landmarks[frame_idx], highlight)
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

    def annotate_frame(self, frame: np.ndarray, landmarks: np.ndarray, highlight_landmarks: list[int] | None = None) -> np.ndarray:
        h, w = frame.shape[:2]
        annotated = frame.copy()
        highlight_set = set(highlight_landmarks or [])

        # Draw skeleton connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            pt1 = (int(landmarks[start_idx, 0] * w), int(landmarks[start_idx, 1] * h))
            pt2 = (int(landmarks[end_idx, 0] * w), int(landmarks[end_idx, 1] * h))
            cv2.line(annotated, pt1, pt2, color=(0, 255, 0), thickness=2)

        # Draw landmark joint points
        for idx in range(len(landmarks)):
            px = int(landmarks[idx, 0] * w)
            py = int(landmarks[idx, 1] * h)
            color = (0, 0, 255) if idx in highlight_set else (0, 255, 0)
            cv2.circle(annotated, (px, py), radius=4, color=color, thickness=-1)

        return annotated