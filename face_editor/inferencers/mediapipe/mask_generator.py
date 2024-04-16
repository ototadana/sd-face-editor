from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
from face_editor.use_cases.mask_generator import MaskGenerator


class MediaPipeMaskGenerator(MaskGenerator):
    def name(self) -> str:
        return "MediaPipe"

    def generate_mask(
        self,
        face_image: np.ndarray,
        face_area_on_image: Tuple[int, int, int, int],
        use_minimal_area: bool,
        mask_size: int,
        use_convex_hull: bool = True,
        dilate_size: int = -1,
        conf: float = 0.01,
        **kwargs,
    ) -> np.ndarray:
        if dilate_size == -1:
            dilate_size = 0 if use_convex_hull else 40

        face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=conf)
        results = face_mesh.process(face_image)

        if not results.multi_face_landmarks:
            return np.zeros(face_image.shape, dtype=np.uint8)

        if use_minimal_area:
            face_image = MaskGenerator.mask_non_face_areas(face_image, face_area_on_image)

        mask = np.zeros(face_image.shape[0:2], dtype=np.uint8)
        for face_landmarks in results.multi_face_landmarks:
            points = []
            for i in range(0, len(face_landmarks.landmark)):
                pt = face_landmarks.landmark[i]
                x, y = int(pt.x * face_image.shape[1]), int(pt.y * face_image.shape[0])
                points.append((x, y))
            if use_convex_hull:
                points = cv2.convexHull(np.array(points))
            cv2.drawContours(mask, [np.array(points)], -1, (255, 255, 255), -1)

        if dilate_size > 0:
            dilate_kernel = np.ones((dilate_size, dilate_size), np.uint8)
            mask = cv2.dilate(mask, dilate_kernel)
            mask = cv2.erode(mask, dilate_kernel)

        if mask_size > 0:
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=mask_size)

        return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
