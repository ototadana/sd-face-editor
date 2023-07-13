from typing import Dict, NamedTuple, Tuple

import numpy as np


class Point(NamedTuple):
    x: int
    y: int


class Landmarks(NamedTuple):
    eye1: Point
    eye2: Point
    nose: Point
    mouth1: Point
    mouth2: Point


class Rect:
    def __init__(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        tag: str = "face",
        landmarks: Landmarks = None,
        attributes: Dict[str, str] = {},
    ) -> None:
        self.tag = tag
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.center = int((right + left) / 2)
        self.middle = int((top + bottom) / 2)
        self.width = right - left
        self.height = bottom - top
        self.size = self.width * self.height
        self.landmarks = landmarks
        self.attributes = attributes

    @classmethod
    def from_ndarray(
        cls,
        face_box: np.ndarray,
        tag: str = "face",
        landmarks: Landmarks = None,
        attributes: Dict[str, str] = {},
    ) -> "Rect":
        left, top, right, bottom, *_ = list(map(int, face_box))
        return cls(left, top, right, bottom, tag, landmarks, attributes)

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
