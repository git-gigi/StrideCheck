import numpy as np
from scipy.signal import find_peaks
from dataclasses import dataclass
from .pose_extractor import VideoMetadata

@dataclass
class GaitCycle:
    start_frame: int
    end_frame: int
    duration_sec: float
    side: str

class GaitSegmenter:
    def __init__(self, landmarks: np.ndarray, angles: dict[str, np.ndarray], metadata: VideoMetadata):
        self.landmarks = landmarks
        self.angles = angles
        self.metadata = metadata

    def segment(self) -> list[GaitCycle]:
        min_distance = int(self.metadata.fps * 0.3)

        peaks_l, _ = find_peaks(self.angles["knee_l"], distance=min_distance, prominence=15)
        peaks_r, _ = find_peaks(self.angles["knee_r"], distance=min_distance, prominence=15)

        print(f"DEBUG - Passi Sinistri: {len(peaks_l)}, Passi Destri: {len(peaks_r)}")

        cycles = []

        for side, peaks in [("left", peaks_l), ("right", peaks_r)]:
            for start, end in zip(peaks[:-1], peaks[1:]):
                cycle = GaitCycle(
                    start_frame=start, 
                    end_frame=end, 
                    duration_sec=(end - start) / self.metadata.fps, 
                    side=side
                    )
                cycles.append(cycle)

        cycles.sort(key=lambda cycle: cycle.start_frame)
            
        return cycles


