from abc import ABC, abstractmethod
from typing import List

from modules.processing import StableDiffusionProcessingImg2Img
from PIL.Image import Image

from scripts.entities.face import Face


class FrameEditor(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def edit(
        self, p: StableDiffusionProcessingImg2Img, faces: List[Face], output_images: List[Image], **kwargs
    ) -> None:
        pass
