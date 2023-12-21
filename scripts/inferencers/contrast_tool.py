from typing import List

import numpy as np
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import ImageEnhance
from PIL.Image import Image

from scripts.entities.face import Face
from scripts.use_cases.frame_editor import FrameEditor
from scripts.use_cases.image_processing_util import add_comment, add_image


class ContrastTool(FrameEditor):
    def name(self) -> str:
        return "Contrast"

    def edit(
        self,
        p: StableDiffusionProcessingImg2Img,
        faces: List[Face],
        output_images: List[Image],
        contrast: float = 1.0,
        **kwargs,
    ) -> None:
        if len(p.init_images) == 0:
            return

        enhancer = ImageEnhance.Contrast(p.init_images[0])
        img_contrasted = enhancer.enhance(contrast)
        p.init_images[0] = img_contrasted

        if output_images is not None:
            output_image = add_comment(np.array(img_contrasted).copy(), f"Contrast: {contrast}%")
            add_image(output_images, output_image)
