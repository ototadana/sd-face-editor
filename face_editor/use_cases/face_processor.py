from abc import ABC, abstractmethod

from face_editor.entities.face import Face
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image


class FaceProcessor(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def process(self, face: Face, p: StableDiffusionProcessingImg2Img, **kwargs) -> Image:
        pass
