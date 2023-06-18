from abc import ABC, abstractmethod
from typing import Tuple

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
