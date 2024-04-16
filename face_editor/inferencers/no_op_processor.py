from face_editor.entities.face import Face
from face_editor.use_cases.face_processor import FaceProcessor
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image


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
