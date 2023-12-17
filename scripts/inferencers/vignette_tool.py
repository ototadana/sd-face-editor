from typing import List

import numpy as np
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image as PILImage
from PIL.Image import Image

from scripts.entities.face import Face
from scripts.inferencers.vignette_mask_generator import VignetteMaskGenerator
from scripts.use_cases.frame_editor import FrameEditor
from scripts.use_cases.image_processing_util import add_comment, add_image


class VignetteTool(FrameEditor):
    def name(self) -> str:
        return "Vignette"

    def edit(
        self,
        p: StableDiffusionProcessingImg2Img,
        faces: List[Face],
        output_images: List[Image],
        sigma: float = 400,
        **kwargs,
    ) -> None:
        if len(p.init_images) == 0:
            return

        mask_generator = VignetteMaskGenerator()

        frame = np.array(p.init_images[0])
        mask = mask_generator.generate_mask(frame, (0, 0, 0, 0), False, sigma, keep_safe_area=False)
        p.init_images[0] = PILImage.fromarray((frame * (mask / 255.0)).astype("uint8"))

        if output_images is not None:
            output_image = add_comment(np.array(p.init_images[0]), f"vignette: sigma: {sigma}")
            add_image(output_images, output_image)
