from typing import List, Union

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL.Image import Image


def rotate_image(image: Image, angle: float) -> Image:
    if image is None:
        return None
    if angle == 0:
        return image
    return PILImage.fromarray(rotate_array(np.array(image), angle))


def rotate_array(image: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def add_comment(image: np.ndarray, comment: str, top: bool = False) -> np.ndarray:
    image = np.copy(image)
    h, _, _ = image.shape
    pos = (10, 48) if top else (10, h - 16)
    cv2.putText(
        image,
        text=comment,
        org=pos,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.2,
        color=(0, 0, 0),
        thickness=10,
    )
    cv2.putText(
        image,
        text=comment,
        org=pos,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.2,
        color=(255, 255, 255),
        thickness=2,
    )
    return image


def add_image(images: List[Image], image: Union[Image, np.ndarray]) -> List[Image]:
    if image is None:
        return images

    if isinstance(image, np.ndarray):
        image = PILImage.fromarray(image)
    images.append(image)
    return images
