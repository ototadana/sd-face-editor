from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image, ImageFilter

from scripts.entities.face import Face
from scripts.use_cases.face_processor import FaceProcessor


class BlurProcessor(FaceProcessor):
    def name(self) -> str:
        return "Blur"

    def process(
        self,
        face: Face,
        p: StableDiffusionProcessingImg2Img,
        radius=20,
        **kwargs,
    ) -> Image:
        return face.image.filter(filter=ImageFilter.GaussianBlur(radius))
