from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np


class MaskGenerator(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def generate_mask(
        self,
        face_image: np.ndarray,
        face_area_on_image: Tuple[int, int, int, int],
        **kwargs,
    ) -> np.ndarray:
        pass

    @staticmethod
    def mask_non_face_areas(image: np.ndarray, face_area_on_image: Tuple[int, int, int, int]) -> np.ndarray:
        left, top, right, bottom = face_area_on_image
        image = image.copy()
        image[:top, :] = 0
        image[bottom:, :] = 0
        image[:, :left] = 0
        image[:, right:] = 0
        return image

    @staticmethod
    def calculate_mask_coverage(mask: np.ndarray):
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        non_black_pixels = np.count_nonzero(gray_mask)
        total_pixels = gray_mask.size
        return non_black_pixels / total_pixels
