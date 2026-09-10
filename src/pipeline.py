from src.pose_extractor import PoseExtractor
from src.angle_calculator import AngleCalculator
from src.feature_extractor import FeatureExtractor
from src.gait_segmenter import GaitSegmenter
from src.savitzky_golay_filter import SavitzkyGolayFilter
from src.analyzer import FormAnalyzer
from src.analyzer import create_analyzer



class AnalysisPipeline:
    def __init__(self, analyzer: FormAnalyzer | None = None):
        self.analyzer = analyzer if analyzer is not None else create_analyzer("heuristic")

    def run(self, video_path: str):
        landmarks, metadata = PoseExtractor().extract(video_path)
        cleaned_landmarks = SavitzkyGolayFilter().apply_filter(landmarks)
        angles = AngleCalculator(cleaned_landmarks).compute_all_angles()
        
        cycles = GaitSegmenter(landmarks=cleaned_landmarks, angles=angles, metadata=metadata).segment()
        features = FeatureExtractor().extract_all(cycles=cycles, angles=angles, landmarks=cleaned_landmarks, fps=metadata.fps)
        result = self.analyzer.analyze(features)

        return result, cleaned_landmarks, metadata, angles, cycles, features