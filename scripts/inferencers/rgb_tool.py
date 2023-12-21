from typing import List

import cv2
import numpy as np
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image as PILImage
from PIL.Image import Image

from scripts.entities.face import Face
from scripts.use_cases.frame_editor import FrameEditor
from scripts.use_cases.image_processing_util import add_comment, add_image


class RgbTool(FrameEditor):
    def name(self) -> str:
        return "RGB"

    def edit(
        self,
        p: StableDiffusionProcessingImg2Img,
        faces: List[Face],
        output_images: List[Image],
        red: float = 1.0,
        green: float = 1.0,
        blue: float = 1.0,
        **kwargs,
    ) -> None:
        if len(p.init_images) == 0:
            return

        frame: Image = p.init_images[0]
        img = np.array(frame).copy()

        red_channel, green_channel, blue_channel = cv2.split(img)

        red_channel = np.clip(red_channel * red, 0, 255).astype(np.uint8)
        green_channel = np.clip(green_channel * green, 0, 255).astype(np.uint8)
        blue_channel = np.clip(blue_channel * blue, 0, 255).astype(np.uint8)

        adjusted_img = cv2.merge([red_channel, green_channel, blue_channel])
        p.init_images[0] = PILImage.fromarray(adjusted_img)

        if output_images is not None:
            output_image = add_comment(adjusted_img, f"R: {red}, G: {green}, B: {blue}")
            add_image(output_images, output_image)
