import cv2
import numpy as np
import pytest
from src.pose_extractor import PoseExtractor, VideoMetadata


class TestPoseExtractor:
    def test_video_metadata_dataclass(self):
        meta = VideoMetadata(
            fps=30.0,
            width=1920,
            height=1080,
            total_frames=300,
            duration_sec=10.0
        )
        assert meta.fps == 30.0
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.total_frames == 300
        assert meta.duration_sec == 10.0

    def test_invalid_video_path_raises_value_error(self):
        extractor = PoseExtractor()
        with pytest.raises(ValueError, match="Cannot open video"):
            extractor.extract("non_existent_path_to_video.mp4")

    def test_extract_from_synthetic_video(self, tmp_path):
        # Create a small 10-frame synthetic MP4 video
        video_file = str(tmp_path / "synthetic_runner.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30.0
        width, height = 320, 240
        out = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
        
        for _ in range(10):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        extractor = PoseExtractor()
        landmarks, metadata = extractor.extract(video_file)

        assert isinstance(metadata, VideoMetadata)
        assert metadata.fps == fps
        assert metadata.width == width
        assert metadata.height == height
        assert metadata.total_frames == 10
        assert np.isclose(metadata.duration_sec, 10 / 30.0)

        assert isinstance(landmarks, np.ndarray)
        assert landmarks.shape == (10, 33, 4)
