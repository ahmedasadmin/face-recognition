from dataclasses import dataclass
import cv2

@dataclass(frozen=True)
class Config:
    DATASET_DIR: str = "../dataset"
    IMGAGE_DIR: str = "../queries"

    IMAGE_MODE: int = cv2.IMREAD_GRAYSCALE
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    RATIO_THRESH: float = 0.75
    THRESHOLD: int = 0.8
    FLANN_ALGORITHM: int = 1
    FLANN_TREES: int = 5
    FLANN_CHECKS: int = 50
    SIFT_N_FEATURES: int = 0
    SIFT_N_OCTAVE_LAYERS: int = 3
    SIFT_CONTRAST_THRESHOLD: float = 0.04
    SIFT_EDGE_THRESHOLD: float = 10
    SIFT_SIGMA: float = 1.6