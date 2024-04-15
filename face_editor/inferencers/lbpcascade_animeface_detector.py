import os
from typing import List, Sequence

import cv2
import numpy as np
from face_editor.entities.rect import Rect
from face_editor.io.util import assets_dir
from face_editor.use_cases.face_detector import FaceDetector
from PIL import Image


class LbpcascadeAnimefaceDetector(FaceDetector):
    def __init__(self) -> None:
        self.cascade_file = os.path.join(assets_dir, "lbpcascade_animeface.xml")

    def name(self):
        return "lbpcascade_animeface"

    def detect_faces(self, image: Image, min_neighbors: int = 5, **kwargs) -> List[Rect]:
        cascade = cv2.CascadeClassifier(self.cascade_file)
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        xywhs = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(24, 24))
        return self.__xywh_to_ltrb(xywhs)

    def __xywh_to_ltrb(self, xywhs: Sequence) -> List[Rect]:
        ltrbs = []
        for xywh in xywhs:
            x, y, w, h = xywh
            ltrbs.append(Rect(x, y, x + w, y + h))
        return ltrbs
