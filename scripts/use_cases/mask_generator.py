from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class MaskGenerator(ABC):
    @abstractmethod
    def generate_mask(
        self,
        face_image: np.ndarray,
        mask_size: int,
        affected_areas: List[str],
        use_minimal_area: bool,
        face_area_on_image: Tuple[int, int, int, int],
    ) -> np.ndarray:
        pass
