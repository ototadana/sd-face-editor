from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image

from scripts.entities.face import Face
from scripts.use_cases.face_processor import FaceProcessor
from scripts.use_cases.image_processing_util import rotate_image


class RotateFaceProcessor(FaceProcessor):
    def name(self) -> str:
        return "Rotate"

    def process(self, face: Face, p: StableDiffusionProcessingImg2Img, angle: float = 0, **kwargs) -> Image:
        return rotate_image(face.image, angle)
