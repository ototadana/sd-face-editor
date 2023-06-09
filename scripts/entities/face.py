import cv2
import numpy as np
from PIL import Image

from scripts.entities.rect import Rect


class Face:
    def __init__(self, entire_image: np.ndarray, face_area: Rect, face_margin: float, face_size: int):
        self.face_area = face_area
        self.center = face_area.center
        left, top, right, bottom = face_area.to_square()

        self.left, self.top, self.right, self.bottom = self.__ensure_margin(
            left, top, right, bottom, entire_image, face_margin)

        self.width = self.right - self.left
        self.height = self.bottom - self.top

        self.image = self.__crop_face_image(entire_image, face_size)
        self.face_area_on_image = self.__get_face_area_on_image(face_size)

    def __get_face_area_on_image(self, face_size: int):
        scaleFactor = face_size / self.width
        return (int((self.face_area.left - self.left) * scaleFactor),
                int((self.face_area.top - self.top) * scaleFactor),
                int((self.face_area.right - self.left) * scaleFactor),
                int((self.face_area.bottom - self.top) * scaleFactor))

    def __crop_face_image(self, entire_image: np.ndarray, face_size: int):
        cropped = entire_image[self.top: self.bottom, self.left: self.right, :]
        return Image.fromarray(
            cv2.resize(cropped, dsize=(face_size, face_size)))

    def __ensure_margin(self, left: int, top: int, right: int, bottom: int, entire_image: np.ndarray, margin: float):
        entire_height, entire_width = entire_image.shape[:2]

        side_length = right - left
        margin = min(min(entire_height, entire_width) /
                     side_length, margin)
        diff = int((side_length * margin - side_length) / 2)

        top = top - diff
        bottom = bottom + diff
        left = left - diff
        right = right + diff

        if top < 0:
            bottom = bottom - top
            top = 0
        if left < 0:
            right = right - left
            left = 0

        if bottom > entire_height:
            top = top - (bottom - entire_height)
            bottom = entire_height
        if right > entire_width:
            left = left - (right - entire_width)
            right = entire_width

        return left, top, right, bottom
