from src.config import Config
import cv2
import numpy as np
from .preprocessing import DataPreprocessor

class SIFTFaceRecognizer:
    def __init__(self, config: Config):
        self.config = config
        self.images = []
        self.labels = []
        self.descriptors = []
        self.preprocessor = DataPreprocessor(image_size=(224, 224))

        self.sift = cv2.SIFT_create(
            nfeatures=config.SIFT_N_FEATURES,
            nOctaveLayers=config.SIFT_N_OCTAVE_LAYERS,
            contrastThreshold=config.SIFT_CONTRAST_THRESHOLD,
            edgeThreshold=config.SIFT_EDGE_THRESHOLD,
            sigma=config.SIFT_SIGMA,
        )

        index_params = dict(
            algorithm=config.FLANN_ALGORITHM,
            trees=config.FLANN_TREES,
        )
        search_params = dict(checks=config.FLANN_CHECKS)

        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def train(self, images, labels):
        self.descriptors = []
        self.labels = []

        for img, label in zip(images, labels):
            roi, _ = self.preprocessor.preprocess(img)
            if roi is None:
                continue

            _, des = self.sift.detectAndCompute(roi, None)
            if des is None:
                continue

            self.descriptors.append(des.astype(np.float32))
            self.labels.append(label)

            
    def predict(self, query_img):
        roi, _ = self.preprocessor.preprocess(query_img)
        if roi is None:
            return None, 0.0

        _, query_des = self.sift.detectAndCompute(roi, None)
        if query_des is None:
            return None, 0.0

        query_des = query_des.astype(np.float32)
        scores = []

        for des, label in zip(self.descriptors, self.labels):
            matches = self.matcher.knnMatch(query_des, des, k=2)
            good = [m for m, n in matches if m.distance < self.config.RATIO_THRESH * n.distance]
            scores.append((label, len(good)))

        scores.sort(key=lambda x: x[1], reverse=True)
        if len(scores) < 2:
            return None, 0.0

        (best_label, best_score), (_, second_score) = scores[:2]
        confidence = best_score / max(second_score, 1)

        print(f"Best={best_label}, conf={confidence:.3f}, matches={best_score}")

        if confidence >= self.config.THRESHOLD:
            return best_label, confidence

        return None, confidence

    def _numpy_img(self, img):
        # img = cv2.imread(img_path, Config.IMAGE_MODE)
        if img is None:
            print(f"[{img}] not valid image") 
        return img