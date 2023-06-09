from abc import ABC, abstractmethod
from typing import List

import numpy as np
from PIL import Image


class MaskGenerator(ABC):
    @abstractmethod
    def generate_mask(self, face_image: Image, mask_size: int, targets: List[str]) -> np.ndarray:
        pass
