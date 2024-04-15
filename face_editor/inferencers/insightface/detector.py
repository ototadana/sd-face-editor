from typing import List

import numpy as np
from face_editor.entities.rect import Landmarks, Point, Rect
from face_editor.use_cases.face_detector import FaceDetector
from insightface.app import FaceAnalysis, common
from PIL import Image


class InsightFaceDetector(FaceDetector):
    def __init__(self):
        self.app = FaceAnalysis(allowed_modules=["detection", "genderage"])
        self.app.prepare(ctx_id=-1)

    def name(self) -> str:
        return "InsightFace"

    def detect_faces(self, image: Image, conf: float = 0.5, **kwargs) -> List[Rect]:
        image_array = np.array(image)

        det_boxes, kpss = self.app.det_model.detect(image_array)

        rects = []

        for box, kps in zip(det_boxes, kpss):
            eye1 = Point(*map(int, kps[0]))
            eye2 = Point(*map(int, kps[1]))
            nose = Point(*map(int, kps[2]))
            mouth2 = Point(*map(int, kps[3]))
            mouth1 = Point(*map(int, kps[4]))

            gender, age = self.app.models["genderage"].get(image_array, common.Face({"bbox": box}))
            gender = "M" if gender == 1 else "F"
            rect = Rect.from_ndarray(
                box,
                "face",
                landmarks=Landmarks(eye1, eye2, nose, mouth1, mouth2),
                attributes={"gender": gender, "age": age},
            )
            rects.append(rect)

        return rects
