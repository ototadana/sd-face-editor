from typing import List, Tuple

import cv2
import numpy as np
from modules import images
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image

from scripts.entities.face import Face
from scripts.entities.option import Option
from scripts.entities.settings import Settings
from scripts.use_cases.frame_editor import FrameEditor
from scripts.use_cases.image_processing_util import add_comment, add_image


class ResizeTool(FrameEditor):
    def name(self) -> str:
        return "Resize"

    def edit(
        self,
        p: StableDiffusionProcessingImg2Img,
        faces: List[Face],
        output_images: List[Image.Image],
        scale: float = None,
        width: int = 512,
        height: int = None,
        upscaler: str = None,
        **kwargs,
    ) -> None:
        if len(p.init_images) == 0:
            return

        if upscaler is None:
            upscaler = Settings.default_upscaler()
            if upscaler == Option.DEFAULT_UPSCALER:
                upscaler = None

        frame: Image.Image = p.init_images[0]
        w, h = self.__get_size(scale, width, height, frame)
        if w == frame.width and h == frame.height:
            return

        p.init_images[0] = images.resize_image(0, frame, w, h, upscaler)
        p.image_mask = Image.fromarray(cv2.resize(np.array(p.image_mask), dsize=(w, h)))

        if output_images is not None:
            output_image = add_comment(np.array(p.init_images[0]), f"resize: {frame.width}x{frame.height} -> {w}x{h}")
            if upscaler is not None:
                output_image = add_comment(output_image, f"upscaler: {upscaler}", top=True)
            add_image(output_images, output_image)

    def __get_size(self, scale: float, width: int, height: int, frame: Image.Image) -> Tuple[int, int]:
        if scale is not None:
            return round(frame.width * scale), round(frame.height * scale)
        if width is not None and height is not None:
            return width, height
        if width is not None:
            return width, round(frame.height * width / frame.width)
        if height is not None:
            return round(frame.width * height / frame.height), height
        return frame.width, frame.height