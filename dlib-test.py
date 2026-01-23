import cv2
import dlib
import numpy as np
import os
import time

# ===================== CONFIG =====================
DATASET_DIR = "../dataset"
SHAPE_PREDICTOR = "../shape_predictor_5_face_landmarks.dat"
FACE_REC_MODEL = "../dlib_face_recognition_resnet_model_v1.dat"
DIST_THRESHOLD = 0.6
CAMERA_ID = 0

# ===================== MODELS =====================
detector = dlib.get_frontal_face_detector()  # HOG (fast)
sp = dlib.shape_predictor(SHAPE_PREDICTOR)
facerec = dlib.face_recognition_model_v1(FACE_REC_MODEL)

# ===================== LOAD KNOWN FACES =====================
known_descriptors = []
known_labels = []

print("[INFO] Loading dataset...")

for person in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        img = dlib.load_rgb_image(img_path)
        dets = detector(img, 1)

        if len(dets) == 0:
            continue

        shape = sp(img, dets[0])
        descriptor = facerec.compute_face_descriptor(img, shape)

        known_descriptors.append(np.array(descriptor))
        known_labels.append(person)

print(f"[INFO] Loaded {len(known_labels)} face samples")

# ===================== CAMERA =====================
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

prev_time = time.perf_counter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(rgb, 0)

    curr_time = time.perf_counter()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    for d in dets:
        shape = sp(rgb, d)
        descriptor = np.array(
            facerec.compute_face_descriptor(rgb, shape)
        )

        distances = np.linalg.norm(
            known_descriptors - descriptor, axis=1
        )

        min_dist = np.min(distances)
        idx = np.argmin(distances)

        if min_dist < DIST_THRESHOLD:
            label = known_labels[idx]
        else:
            label = "Unknown"

        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"{label} ({min_dist:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    cv2.putText(frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 0, 0), 2)

    cv2.imshow("Face Recognition (dlib)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
