import cv2
import numpy as np
from face_editor.entities.face import Face
from face_editor.use_cases.face_processor import FaceProcessor
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (0, 128, 128),
]


def color_generator(colors):
    while True:
        for color in colors:
            yield color


color_iter = color_generator(colors)


class DebugProcessor(FaceProcessor):
    def name(self) -> str:
        return "Debug"

    def process(
        self,
        face: Face,
        p: StableDiffusionProcessingImg2Img,
        **kwargs,
    ) -> Image:
        image = np.array(face.image)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), next(color_iter), -1)
        l, t, r, b = face.face_area_on_image
        cv2.rectangle(overlay, (l, t), (r, b), (0, 0, 0), 10)
        if face.landmarks_on_image is not None:
            for landmark in face.landmarks_on_image:
                cv2.circle(overlay, (int(landmark.x), int(landmark.y)), 6, (0, 0, 0), 10)
        alpha = 0.3
        output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        return Image.fromarray(output)
