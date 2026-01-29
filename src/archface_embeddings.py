import torch
from insightface.app import FaceAnalysis

class ArchFaceEmbedder:

    def __init__(self, model_name = "buffalo_s"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = None
        self.features_model = FaceAnalysis(
            name = model_name,
            provider = [
                'CUDAExecutionProvider' 
                if self.device == 'cuda'
                else 'CPUExecutionProvider'
            ]
        )

        self.features_model.prepare(ctx_id = 0 if self.device == 'cuda' else -1)

    def get_embedding(self, rgb_img):
        faces = self.features_model.get(rgb_img)
        return faces
 
