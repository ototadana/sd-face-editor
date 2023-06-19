from abc import ABC, abstractmethod

from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image

from scripts.entities.face import Face


class FaceProcessor(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def process(self, face: Face, p: StableDiffusionProcessingImg2Img, **kwargs) -> Image:
        pass
