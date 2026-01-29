 
import cv2 as cv
import os 
from abc import ABC, abstractmethod 
import pickle
import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def frames(self):
        pass
    @abstractmethod
    def release(self):
        pass

class VideoSource(FrameSource):
    def __init__(self, source):
        self.cap = cv.VideoCapture(source)
    
    def frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break 
        
            yield frame
    def release(self):
        self.cap.release()
    
            
class ImageFolderSource(FrameSource):
    def __init__(self, folder):
        self.images = [os.path.join(folder, f)
                       for f in os.listdir(folder)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    def frames(self):
        for img in self.images:
            frame = cv.imread(img, cv.IMREAD_COLOR)
            if frame is not None:
                yield frame
    
    def release(self):
        pass

class FaceRecognizer:
    def __init__(self, recognizer, preprocessor):
        self.recognizer = recognizer
        self.preprocessor = preprocessor
    
    def recognize(self, frame):
        frame = self.preprocessor.preprocess(frame)
        if frame is None:
            return [] 
        

        faces = self.recognizer.predict(frame, "yolov8n-face.pt")
        return faces 
    

    
class Renderer:
    @staticmethod
    def draw(frame, bbox, label, conf):
        
        if bbox is None:
            return frame
        x1, y1, x2, y2 = bbox
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        
        return frame
    @staticmethod
    def save_image(frame, label, folder="results"):
        if label is None:
            return False

        os.makedirs(folder, exist_ok=True)

        i = 0
        while True:
            image_path = os.path.join(folder, f"{label}_{i}.png")
            if not os.path.exists(image_path):
                break
            i += 1

        return cv.imwrite(image_path, frame)
    

class ModelArtifactsManger:
    def __init__(self):
        self.class_means = {}
    
    def save_embeddings(self, features , labels, file_name="embeddings.pkl"):
        data = {
            "features": np.asarray(features),
            "labels": np.asarray(labels)
        }
        with open(file_name, "wb") as f:
            pickle.dump(data, f)
    
    def save_label_encoder(self, encoder, path="label_encoder.pkl"):
        with open(path, "wb") as f:
            pickle.dump(encoder, f)
    def load_label_encoder(self, path="label_encoder.pkl"):

        with open(path, "rb") as f:
            return pickle.load(f)
    def build(self, X, y):
        for cls in np.unique(y):
            self.class_means[cls] = X[y == cls].mean(axis=0)

    def save(self, path="prototypes.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.class_means, f)

    @staticmethod
    def load(path="prototypes.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
