from typing import List

import modules.shared as shared
import torch
from facexlib.detection import init_detection_model, retinaface
from PIL import Image

from scripts.entities.rect import Rect
from scripts.use_cases.face_detector import FaceDetector


class RetinafaceDetector(FaceDetector):
    def __init__(self) -> None:
        if hasattr(retinaface, "device"):
            retinaface.device = shared.device
        self.detection_model = init_detection_model("retinaface_resnet50", device=shared.device)

    def name(self):
        return "RetinaFace"

    def detect_faces(self, image: Image, confidence: float, **kwargs) -> List[Rect]:
        with torch.no_grad():
            face_boxes, _ = self.detection_model.align_multi(image, confidence)

        faces = []
        for face_box in face_boxes:
            faces.append(Rect.from_ndarray(face_box))
        return faces
