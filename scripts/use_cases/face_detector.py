from abc import ABC, abstractmethod
from typing import List

from PIL import Image

from scripts.entities.rect import Rect


class FaceDetector(ABC):

    @abstractmethod
    def detect_faces(self, image: Image, confidence: float) -> List[Rect]:
        pass
