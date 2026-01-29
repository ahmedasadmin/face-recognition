from insightface.app import FaceAnalysis
import torch 
import os 
import numpy as np
import cv2 as cv
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.svm import SVC
import pickle
from sklearn.preprocessing import Normalizer
from src.utils import ModelArtifactsManger
from src.data_loader import DataLoader
from src.svm_classifier import SVMClassifier
from src.archface_embeddings import ArchFaceEmbedder





if __name__ =='__main__':

    em_model = ArchFaceEmbedder()
    svm_obj = SVMClassifier()
    util_obj = ModelArtifactsManger()
    loader_obj = DataLoader("/home/ahmed/coding/face-recognition/staff-face-recog/dataset", 
                               em_model
                               ) 
    features, labels = loader_obj.load_data()
    print(f"[INFO] Features: {len(features)}, Labels{len(labels)}")

    features = Normalizer(norm="l2").fit_transform(features)
    util_obj.save_embeddings(features, labels, "models/arcface_embeddings.pkl")
    print(f"[INFO] ArcFace embeddings save successfully.")

    
    util_obj.build(features, labels)
    util_obj.save("models/arcface_prototypes.pkl")
    print(f"[INFO] Encoded labels save successfully.")

    util_obj.save_label_encoder(loader_obj.encoder, "models/label_encoder.pkl")
    print(f"[INFO] Encoded labels save successfully.")

    svm_obj.train(features, labels)
   
    svm_obj.save("models/svm_arcface.pkl")
    print(f"[INFO] SVM model saved successfully.")
