import cv2
import pathlib
from typing import List, Tuple, Optional
import logging
from src.config import Config
# Optional: configure once in your project
logging.basicConfig(level=logging.WARNING)


class BaseImageLoader:
    """Shared logic for all image loaders."""

    def __init__(self, config:Config, logger: Optional[logging.Logger] = None):
        self.image_mode = Config.IMAGE_MODE 
        self.extensions = Config.SUPPORTED_EXTENSIONS
        self.logger = logger or logging.getLogger(__name__)
        self._images: List[cv2.Mat] = []  # Protected, for internal use

    def _is_valid_image_file(self, path: pathlib.Path) -> bool:
        return path.is_file() and path.suffix.lower() in self.extensions

    def _read_image(self, path: pathlib.Path):
        img_path_str = str(path)
        img = cv2.imread(img_path_str, self.image_mode)
        if img is None:
            self.logger.warning(f"Failed to read image: {img_path_str}")
        return img

    def get_images(self) -> List[cv2.Mat]:
        """Return the loaded images (read-only view)."""
        return self._images.copy()


class ImageDatasetLoader(BaseImageLoader):
    """
    Loads images for classification tasks.
    Folder structure: root_dir / person_name / image_files
    Returns: list of images, list of labels (person_name)
    """
    
    def load(self, root_dir: str= Config.DATASET_DIR) -> Tuple[List[cv2.Mat], List[str]]:
        root = pathlib.Path(root_dir)
        if not root.is_dir():
            raise ValueError(f"Root directory not found: {root_dir}")

        images: List[cv2.Mat] = []
        labels: List[str] = []

        for person_dir in root.iterdir():
            if not person_dir.is_dir():
                continue

            label = person_dir.name
            for img_path in person_dir.iterdir():
                if not self._is_valid_image_file(img_path):
                    continue

                img = self._read_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(label)

        self._images = images  # Keep internal copy if needed
        return images, labels


class ImageLoader(BaseImageLoader):
    """
    Loads all images from a single folder (no labels).
    Folder structure: root_dir / image_files
    """
    
    def load(self, root_dir: str) -> List[cv2.Mat]:
        root = pathlib.Path(root_dir)
        if not root.is_dir():
            raise ValueError(f"Root directory not found: {root_dir}")

        images: List[cv2.Mat] = []

        for img_path in root.glob("*"):
            if not self._is_valid_image_file(img_path):
                continue

            img = self._read_image(img_path)
            if img is not None:
                images.append(img)

        self._images = images
        return images