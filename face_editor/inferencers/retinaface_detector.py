from typing import List

import modules.shared as shared
import torch
from face_editor.entities.rect import Landmarks, Point, Rect
from face_editor.use_cases.face_detector import FaceDetector
from facexlib.detection import init_detection_model, retinaface
from PIL import Image


class RetinafaceDetector(FaceDetector):
    def __init__(self) -> None:
        if hasattr(retinaface, "device"):
            retinaface.device = shared.device
        self.detection_model = init_detection_model("retinaface_resnet50", device=shared.device)

    def name(self):
        return "RetinaFace"

    def detect_faces(self, image: Image, confidence: float, **kwargs) -> List[Rect]:
        with torch.no_grad():
            boxes_landmarks = self.detection_model.detect_faces(image, confidence)

        faces = []
        for box_landmark in boxes_landmarks:
            face_box = box_landmark[:5]
            landmark = box_landmark[5:]
            face = Rect.from_ndarray(face_box)

            eye1 = Point(int(landmark[0]), int(landmark[1]))
            eye2 = Point(int(landmark[2]), int(landmark[3]))
            nose = Point(int(landmark[4]), int(landmark[5]))
            mouth2 = Point(int(landmark[6]), int(landmark[7]))
            mouth1 = Point(int(landmark[8]), int(landmark[9]))

            face.landmarks = Landmarks(eye1, eye2, nose, mouth1, mouth2)
            faces.append(face)

        return faces
