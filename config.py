from dataclasses import dataclass
import cv2

@dataclass(frozen=True)
class Config:
    DATASET_DIR: str     = "/home/ahmed/coding/face-recognition/staff-face-recog/dataset"
    EMBEDDING_PATH:str   = "models/arcface_embeddings.pkl"
    PROTOTYPE_PATH:str   = "models/arcface_prototypes.pkl"
    LABELS_PATH:str      ="models/label_encoder.pkl"
    IMAGE_MODE: int      = cv2.IMREAD_GRAYSCALE
    SIM_THRESHOLD: int   = 0.2
    IMAGE_SIZE:tuple     = (224, 224) 
    PROB_THRESHOLD:float = 0.65


    