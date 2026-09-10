from dataclasses import dataclass
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
import numpy as np


@dataclass
class VideoMetadata:
    fps: float
    width: int
    height: int
    total_frames: int
    duration_sec: float


class PoseExtractor:
    """Wraps MediaPipe Tasks PoseLandmarker to extract 3D landmarks from video."""

    def __init__(self, model_path: str = "models/pose_landmarker_heavy.task"):
        self.model_path = model_path
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def extract(self, video_path: str) -> tuple[np.ndarray, VideoMetadata]:
        cap = self._open_video(video_path)
        metadata = self._extract_metadata(cap)
        landmarks = self._process_all_frames(cap, metadata.fps)
        cap.release()
        return landmarks, metadata

    def _open_video(self, video_path: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        return cap

    def _extract_metadata(self, cap: cv2.VideoCapture) -> VideoMetadata:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoMetadata(
            fps=fps,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            total_frames=total_frames,
            duration_sec=total_frames / fps if fps > 0 else 0.0,
        )

    def _process_all_frames(self, cap: cv2.VideoCapture, fps: float) -> np.ndarray:
        all_landmarks = []
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            timestamp_ms = int(frame_idx * 1000.0 / fps) if fps > 0 else frame_idx * 33
            all_landmarks.append(self._extract_single_frame(frame, timestamp_ms))
            frame_idx += 1

        if not all_landmarks:
            raise ValueError("No frames could be processed from the video.")

        return np.array(all_landmarks)

    def _extract_single_frame(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            return np.array([
                [lm.x, lm.y, lm.z, getattr(lm, "visibility", getattr(lm, "presence", 1.0))]
                for lm in result.pose_landmarks[0]
            ])

        return np.zeros((33, 4))
