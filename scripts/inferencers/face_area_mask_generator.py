from typing import Tuple

import numpy as np

from scripts.use_cases.mask_generator import MaskGenerator


class FaceAreaMaskGenerator(MaskGenerator):
    def name(self):
        return "Rect"

    def generate_mask(
        self,
        face_image: np.ndarray,
        face_area_on_image: Tuple[int, int, int, int],
        **kwargs,
    ) -> np.ndarray:
        height, width = face_image.shape[:2]
        img = np.ones((height, width, 3), np.uint8) * 255
        return FaceAreaMaskGenerator.mask_non_face_areas(img, face_area_on_image)

    @staticmethod
    def mask_non_face_areas(image: np.ndarray, face_area_on_image: Tuple[int, int, int, int]) -> np.ndarray:
        left, top, right, bottom = face_area_on_image
        image = image.copy()
        image[:top, :] = 0
        image[bottom:, :] = 0
        image[:, :left] = 0
        image[:, right:] = 0
        return image
