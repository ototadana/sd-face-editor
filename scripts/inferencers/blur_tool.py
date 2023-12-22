from typing import List

import numpy
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import ImageFilter
from PIL.Image import Image

from scripts.entities.face import Face
from scripts.use_cases.frame_editor import FrameEditor
from scripts.use_cases.image_processing_util import add_comment, add_image


class BlurTool(FrameEditor):
    def name(self) -> str:
        return "Blur"

    def edit(
        self,
        p: StableDiffusionProcessingImg2Img,
        faces: List[Face],
        output_images: List[Image],
        radius: float = 2,
        **kwargs,
    ) -> None:
        if len(p.init_images) == 0:
            return

        p.init_images[0] = p.init_images[0].filter(filter=ImageFilter.GaussianBlur(radius))

        if output_images is not None:
            output_image = add_comment(numpy.array(p.init_images[0]).copy(), f"Blur: {radius}")
            add_image(output_images, output_image)
