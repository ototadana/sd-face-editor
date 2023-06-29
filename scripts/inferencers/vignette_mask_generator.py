from typing import Tuple

import cv2
import numpy as np

from scripts.use_cases.mask_generator import MaskGenerator


class VignetteMaskGenerator(MaskGenerator):
    def name(self):
        return "Vignette"

    def generate_mask(
        self,
        face_image: np.ndarray,
        face_area_on_image: Tuple[int, int, int, int],
        sigma: float = 120,
        keep_safe_area: bool = False,
        **kwargs,
    ) -> np.ndarray:
        (left, top, right, bottom) = face_area_on_image
        w, h = right - left, bottom - top
        mask = np.zeros((face_image.shape[0], face_image.shape[1]), dtype=np.uint8)
        mask[top : top + h, left : left + w] = 255

        Y = np.linspace(0, h, h, endpoint=False)
        X = np.linspace(0, w, w, endpoint=False)
        Y, X = np.meshgrid(Y, X)
        Y -= h / 2
        X -= w / 2

        gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        gaussian_mask = np.uint8(255 * gaussian.T)
        mask[top : top + h, left : left + w] = gaussian_mask
        if keep_safe_area:
            mask = cv2.ellipse(mask, ((left + right) // 2, (top + bottom) // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return mask
