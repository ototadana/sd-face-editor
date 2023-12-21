from typing import List

import cv2
import numpy as np
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image as PILImage
from PIL.Image import Image

from scripts.entities.face import Face
from scripts.use_cases.frame_editor import FrameEditor
from scripts.use_cases.image_processing_util import add_comment, add_image


class HslTool(FrameEditor):
    def name(self) -> str:
        return "HSL"

    def edit(
        self,
        p: StableDiffusionProcessingImg2Img,
        faces: List[Face],
        output_images: List[Image],
        hue: int = 0,
        saturation: float = 1.0,
        lightness: float = 1.0,
        **kwargs,
    ) -> None:
        if len(p.init_images) == 0:
            return

        frame: Image = p.init_images[0]
        img = np.array(frame).copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        hue_adjust = hue / 360

        h, l, s = cv2.split(img)

        h = np.mod(h + hue_adjust * 180, 180).astype(np.uint8)
        s = np.clip(s * saturation, 0, 255).astype(np.uint8)
        l = np.clip(l * lightness, 0, 255).astype(np.uint8)

        adjusted_img = cv2.merge([h, l, s])
        adjusted_img = cv2.cvtColor(adjusted_img, cv2.COLOR_HLS2RGB)
        p.init_images[0] = PILImage.fromarray(adjusted_img)

        if output_images is not None:
            output_image = add_comment(adjusted_img, f"H: {hue}, S: {saturation}, L: {lightness}")
            add_image(output_images, output_image)
