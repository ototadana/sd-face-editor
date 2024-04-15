from abc import ABC, abstractmethod
from typing import List

from face_editor.entities.rect import Rect
from PIL import Image


class FaceDetector(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def detect_faces(self, image: Image, **kwargs) -> List[Rect]:
        pass
