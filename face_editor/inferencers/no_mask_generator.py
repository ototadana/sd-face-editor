from typing import Tuple

import numpy as np
from face_editor.use_cases.mask_generator import MaskGenerator


class NoMaskGenerator(MaskGenerator):
    def name(self):
        return "NoMask"

    def generate_mask(
        self,
        face_image: np.ndarray,
        face_area_on_image: Tuple[int, int, int, int],
        **kwargs,
    ) -> np.ndarray:
        height, width = face_image.shape[:2]
        return np.ones((height, width, 3), np.uint8) * 255
