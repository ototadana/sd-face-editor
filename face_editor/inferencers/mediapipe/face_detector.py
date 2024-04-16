from typing import List

import mediapipe as mp
import numpy as np
from face_editor.entities.rect import Landmarks, Point, Rect
from face_editor.use_cases.face_detector import FaceDetector
from PIL import Image


class MediaPipeFaceDetector(FaceDetector):
    def name(self) -> str:
        return "MediaPipe"

    def detect_faces(self, image: Image, conf: float = 0.01, **kwargs) -> List[Rect]:
        face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=conf)
        results = face_detection.process(np.array(image.convert("RGB")))

        width = image.width
        height = image.height
        rects: List[Rect] = []

        if not results.detections:
            return rects

        for d in results.detections:
            relative_box = d.location_data.relative_bounding_box
            left = int(relative_box.xmin * width)
            top = int(relative_box.ymin * height)
            right = int(left + (relative_box.width * width))
            bottom = int(top + (relative_box.height * height))

            keypoints = d.location_data.relative_keypoints

            eye1 = Point(int(keypoints[0].x * width), int(keypoints[0].y * height))
            eye2 = Point(int(keypoints[1].x * width), int(keypoints[1].y * height))
            nose = Point(int(keypoints[2].x * width), int(keypoints[2].y * height))
            mouth = Point(int(keypoints[3].x * width), int(keypoints[3].y * height))

            rects.append(Rect(left, top, right, bottom, landmarks=Landmarks(eye1, eye2, nose, mouth, mouth)))

        return rects
