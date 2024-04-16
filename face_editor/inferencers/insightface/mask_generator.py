from typing import Tuple

import cv2
import numpy as np
from face_editor.use_cases.mask_generator import MaskGenerator
from insightface.app import FaceAnalysis, common


class InsightFaceMaskGenerator(MaskGenerator):
    def __init__(self):
        self.app = FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"])
        self.app.prepare(ctx_id=-1)

    def name(self):
        return "InsightFace"

    def generate_mask(
        self,
        face_image: np.ndarray,
        face_area_on_image: Tuple[int, int, int, int],
        use_minimal_area: bool,
        mask_size: int,
        use_convex_hull: bool = True,
        dilate_size: int = -1,
        **kwargs,
    ) -> np.ndarray:
        if dilate_size == -1:
            dilate_size = 0 if use_convex_hull else 40

        face = common.Face({"bbox": np.array(face_area_on_image)})
        landmarks = self.app.models["landmark_2d_106"].get(face_image, face)

        if use_minimal_area:
            face_image = MaskGenerator.mask_non_face_areas(face_image, face_area_on_image)

        mask = np.zeros(face_image.shape[0:2], dtype=np.uint8)
        points = [(int(landmark[0]), int(landmark[1])) for landmark in landmarks]
        if use_convex_hull:
            points = cv2.convexHull(np.array(points))
        cv2.drawContours(mask, [np.array(points)], -1, (255, 255, 255), -1)

        if dilate_size > 0:
            kernel = np.ones((dilate_size, dilate_size), np.uint8)
            mask = cv2.dilate(mask, kernel)
            mask = cv2.erode(mask, kernel)

        if mask_size > 0:
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=mask_size)

        return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
