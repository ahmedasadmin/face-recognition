import cv2
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import Normalizer

from src.utils import FrameSource, VideoSource, ImageFolderSource, Renderer
from src.archface_embeddings import ArchFaceEmbedder


class FaceRecognizer:
    def __init__(
        self,
        embedder,
        svm_model,
        label_encoder,
        prototypes,
        sim_threshold=0.2,
        prob_threshold=0.65
    ):
        self.embedder = embedder
        self.svm = svm_model
        self.encoder = label_encoder
        self.prototypes = prototypes
        self.sim_th = sim_threshold
        self.prob_th = prob_threshold
        self.norm = Normalizer(norm="l2")

    def recognize(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.embedder.get_embedding(rgb)

        results = []

        for face in faces:
            emb = self.norm.transform(face.embedding.reshape(1, -1))[0]

            probs = self.svm.predict_proba([emb])[0]
            cls_id = int(np.argmax(probs))
            prob = float(probs[cls_id])

            proto = self.prototypes[cls_id]
            sim = float(np.dot(emb, proto))

            if prob < self.prob_th or sim < self.sim_th:
                label = "Unknown"
                conf = max(prob, sim)
            else:
                label = self.encoder.inverse_transform([cls_id])[0]
                conf = prob

            results.append({
                "bbox": face.bbox.astype(int),
                "label": label,
                "conf": conf
            })

        return results



class FaceRecognitionApp:
    def __init__(self, source: FrameSource, recognizer: FaceRecognizer):
        self.source = source
        self.recognizer = recognizer

    def run(self):
        for frame in self.source.frames():
            faces = self.recognizer.recognize(frame)

            h, w = frame.shape[:2]

            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if face["conf"] > 0.5:
                    Renderer.draw(
                        frame,
                        (x1, y1, x2, y2),
                        face["label"],
                        face["conf"]
                    )

            cv2.imshow("Face Recognition (ArcFace)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.source.release()
        cv2.destroyAllWindows()


def main(stream_type="VIDEO"):
 
    embedder = ArchFaceEmbedder()

    # Load trained artifacts
    svm_model = joblib.load("models/svm_arcface.pkl")

    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("models/arcface_prototypes.pkl", "rb") as f:
        prototypes = pickle.load(f)

    recognizer = FaceRecognizer(
        embedder=embedder,
        svm_model=svm_model,
        label_encoder=label_encoder,
        prototypes=prototypes
    )

    if stream_type == "VIDEO":
        source = VideoSource(0)
    else:
        source = ImageFolderSource("test")

    app = FaceRecognitionApp(source, recognizer)
    app.run()


if __name__ == "__main__":
    main("VIDEO")
