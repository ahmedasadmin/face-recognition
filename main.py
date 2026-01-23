import cv2
import dlib
import time
from src.data_loader import ImageDatasetLoader
from src.recognizer import SIFTFaceRecognizer
from src.preprocessing import DataPreprocessor
from src.config import Config

if __name__ == "__main__":
    config = Config()
    detector = dlib.get_frontal_face_detector()

    dataset = ImageDatasetLoader(config)
    images, labels = dataset.load()

    recognizer = SIFTFaceRecognizer(config)
    recognizer.train(images, labels)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Cannot open camera")

    # ---- FPS init (ONCE) ----
    t_prev = time.perf_counter()
    fps = 0.0
    alpha = 0.9
    # ------------------------
    preprocessor = DataPreprocessor(image_size=(224, 224))
    while True:
        ret, query_img = cap.read()
        if not ret:
            break
        face, bbox = preprocessor.preprocess(query_img)

        label = "Unknown face"
        if face is not None:

            label, score = recognizer.predict(face)
            if label is None:
                label = "Unknown face"

            x1, y1, x2, y2 = bbox
            # ---- FPS (end-to-end) ----
            t_now = time.perf_counter()
            inst_fps = 1.0 / (t_now - t_prev)
            fps = alpha * fps + (1 - alpha) * inst_fps
            t_prev = t_now
            # --------------------------
            cv2.rectangle(query_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(query_img, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.putText(query_img, f"FPS: {fps:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("frame", query_img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
