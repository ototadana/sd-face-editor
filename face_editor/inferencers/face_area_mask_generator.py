from typing import Tuple

import numpy as np
from face_editor.use_cases.mask_generator import MaskGenerator


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
        return MaskGenerator.mask_non_face_areas(img, face_area_on_image)
