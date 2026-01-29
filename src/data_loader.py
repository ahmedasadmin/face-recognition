   
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2 as cv


class DataLoader:
    def __init__(self, folder, embeddings_model):
        """
        take the dataset folder and return Features (Embeddings) and Labels (Names)

        """
        self.folder = folder
        self.embeddings_model = embeddings_model
        self.features = []
        self.labels = []
    def load_data(self):
        try:
                
            for label in os.listdir(self.folder):
                class_path = os.path.join(self.folder, label)
                if not os.path.isdir(class_path):
                    continue
                for file in os.listdir(class_path):
                    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                            continue
                    img_full_path = os.path.join(class_path, file)
                    img = cv.imread(img_full_path)
                    rgb_img = self.process(img)
                    faces = self.embeddings_model.get_embedding(np.asarray(rgb_img))
                    if len(faces) == 0:
                        print(f"[WARNING] No faces detected in {img_full_path}")
                        continue

                    self.features.append(faces[0].embedding)
                    self.labels.append(label)
                    
                
        except Exception as e:
            raise RuntimeError("[ERROR] Can not load data (features, lables)")
        self.labels = self._encode_label(self.labels)
        return np.asarray(self.features), np.asarray(self.labels) 
    
    def _encode_label(self, labels):
        """
        encoded the output 'label' into numerical representation appropriate for the model 
        e.g. [ahmed, tony stark, ...] --> [0, 1, ...]
        """
        self.encoder = LabelEncoder()
        return self.encoder.fit_transform(labels)

    
    def process(self, img):
        """
         process: resize and color coversion

        """
        
        rgb_image  = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return rgb_image
    
