import os
from typing import List

import cv2
import modules.scripts as scripts
import numpy as np
from PIL import Image

from scripts.entities.rect import Rect
from scripts.use_cases.face_detector import FaceDetector


class LbpcascadeAnimefaceDetector(FaceDetector):
    def __init__(self) -> None:
        self.cascade_file = os.path.join(scripts.basedir(), "assets", "lbpcascade_animeface.xml")
        if not os.path.isfile(self.cascade_file):
            self.cascade_file = os.path.join(
                scripts.basedir(), "extensions", "sd-face-editor", "assets", "lbpcascade_animeface.xml"
            )
            if not os.path.isfile(self.cascade_file):
                raise RuntimeError(f"not found:{self.cascade_file}")

    def name(self):
        return "lbpcascade_animeface"

    def detect_faces(self, image: Image, confidence: float) -> List[Rect]:
        cascade = cv2.CascadeClassifier(self.cascade_file)
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        minNeighbors = self.__confidence_to_minNeighbors(confidence)
        xywhs = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=minNeighbors, minSize=(24, 24))
        return self.__xywh_to_ltrb(xywhs)

    def __xywh_to_ltrb(self, xywhs: list) -> List[Rect]:
        ltrbs = []
        for xywh in xywhs:
            x, y, w, h = xywh
            ltrbs.append(Rect(x, y, x + w, y + h))
        return ltrbs

    def __confidence_to_minNeighbors(self, confidence: float) -> int:
        minNeighbors = int((confidence - 0.7) * 20)
        return minNeighbors if minNeighbors > 0 else 1
