from face_editor.entities.face import Face
from face_editor.use_cases.face_processor import FaceProcessor
from face_editor.use_cases.image_processing_util import rotate_image
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image


class RotateFaceProcessor(FaceProcessor):
    def name(self) -> str:
        return "Rotate"

    def process(self, face: Face, p: StableDiffusionProcessingImg2Img, angle: float = 0, **kwargs) -> Image:
        return rotate_image(face.image, angle)
