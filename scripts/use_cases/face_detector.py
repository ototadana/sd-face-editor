from abc import ABC, abstractmethod
from typing import List

from PIL.Image import Image

from scripts.entities.rect import Rect


class FaceDetector(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def detect_faces(self, image: Image, **kwargs) -> List[Rect]:
        pass
