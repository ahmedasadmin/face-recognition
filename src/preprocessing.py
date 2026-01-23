import cv2 
import dlib


class DataPreprocessor:
    def __init__(
        self,
        image_size=(128, 128),
        upsample=0,
        use_gray=True,
        apply_hist = True
    ):
        self.image_size = image_size
        self.upsample = upsample
        self.use_gray = use_gray
        self.apply_hist = apply_hist
        # HOG-based detector (fast, no GPU)
        self.detector = dlib.get_frontal_face_detector()

    def preprocess(self, image):
        """
        Input:
            image: BGR image from OpenCV
        Output:
            face_roi: processed face image or None
            bbox: (x1, y1, x2, y2) or None
        """

        if image is None:
            return None, None
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image 
        faces = self.detector(gray, self.upsample)
        if len(faces) == 0:
            return None, None

        # Take largest face
        face = max(faces, key=lambda r: r.width() * r.height())

        x1, y1 =  face.left(),  face.top()
        x2, y2 = face.right(), face.bottom()

        face_roi = gray[y1:y2, x1:x2]

        if face_roi.size == 0:
            return None, None

        face_roi = cv2.resize(face_roi, self.image_size)
        cv2.imshow("Frame", face_roi)
        cv2.waitKey(0)
        if self.apply_hist:
            face_roi = self._apply_hist(face_roi)

        return face_roi, (x1, y1, x2, y2)
    def _apply_hist(self, image):
        
         return cv2.equalizeHist(image)