from typing import List, Tuple

import numpy as np
from face_editor.inferencers.face_area_mask_generator import FaceAreaMaskGenerator
from face_editor.use_cases.mask_generator import MaskGenerator
from PIL import Image, ImageDraw


class EllipseMaskGenerator(MaskGenerator):
    def name(self):
        return "Ellipse"

    def generate_mask(
        self,
        face_image: np.ndarray,
        face_area_on_image: Tuple[int, int, int, int],
        use_minimal_area: bool,
        **kwargs,
    ) -> np.ndarray:
        if use_minimal_area:
            face_image = FaceAreaMaskGenerator.mask_non_face_areas(face_image, face_area_on_image)

        height, width = face_image.shape[:2]
        img = Image.new("RGB", (width, height), (255, 255, 255))
        mask = Image.new("L", (width, height))
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.ellipse(self.__get_box(face_area_on_image), fill=255)
        black_img = Image.new("RGB", (width, height))
        face_image = Image.composite(img, black_img, mask)
        return np.array(face_image)

    def __get_box(self, face_area_on_image: Tuple[int, int, int, int]) -> List[int]:
        left, top, right, bottom = face_area_on_image
        factor = 1.2
        width = right - left
        height = bottom - top
        a_width = int(width * factor)
        a_height = int(height * factor)
        left = left - int((a_width - width) / 2)
        top = top - int((a_height - height) / 2)
        return [left, top, a_width, a_height]
