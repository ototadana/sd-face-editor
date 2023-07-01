from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image

from scripts.entities.face import Face
from scripts.use_cases.face_processor import FaceProcessor


class NoOpProcessor(FaceProcessor):
    def name(self) -> str:
        return "NoOp"

    def process(
        self,
        face: Face,
        p: StableDiffusionProcessingImg2Img,
        **kwargs,
    ) -> Image:
        return None
