from typing import Tuple

import numpy as np


class Rect:
    def __init__(self, left: int, top: int, right: int, bottom: int, tag: str = "face") -> None:
        self.tag = tag
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.center = right - int((right - left) / 2)

    @classmethod
    def from_ndarray(cls, face_box: np.ndarray, tag: str = "face") -> "Rect":
        left, top, right, bottom, *_ = list(map(int, face_box))
        return cls(left, top, right, bottom, tag)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    def to_square(self):
        left, top, right, bottom = self.to_tuple()

        width = right - left
        height = bottom - top

        if width % 2 == 1:
            right = right + 1
            width = width + 1
        if height % 2 == 1:
            bottom = bottom + 1
            height = height + 1

        diff = int(abs(width - height) / 2)
        if width > height:
            top = top - diff
            bottom = bottom + diff
        else:
            left = left - diff
            right = right + diff

        return left, top, right, bottom
