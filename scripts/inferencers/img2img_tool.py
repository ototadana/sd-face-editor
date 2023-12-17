from typing import List

import modules.scripts as scripts
import numpy
from modules.processing import StableDiffusionProcessingImg2Img, process_images
from PIL.Image import Image

from scripts.entities.face import Face
from scripts.inferencers.temp_settings import temp_attr, temp_dict
from scripts.use_cases.frame_editor import FrameEditor
from scripts.use_cases.image_processing_util import add_comment, add_image
from scripts.use_cases.mask_generator import MaskGenerator


class Img2ImgTool(FrameEditor):
    def name(self) -> str:
        return "img2img"

    def edit(
        self,
        p: StableDiffusionProcessingImg2Img,
        faces: List[Face],
        output_images: List[Image],
        pp: str = "",
        np: str = "",
        use_refiner_model_only=False,
        strength=-1,
        no_mask=False,
        **kwargs,
    ) -> None:
        if strength == -1:
            strength = p.denoising_strength if p.denoising_strength is not None and p.denoising_strength > 0 else 0.4

        if strength == 0:
            return

        if p.scripts is None:
            p.scripts = scripts.ScriptRunner()

        with temp_attr(
            p,
            denoising_strength=strength,
            prompt=pp if len(pp) > 0 else p.prompt,
            negative_prompt=np if len(np) > 0 else p.negative_prompt,
            refiner_switch_at=0 if use_refiner_model_only else p.refiner_switch_at,
            image_mask=None if no_mask else p.image_mask,
        ):
            if (
                getattr(p, "enable_hr", False)
                and hasattr(p, "hr_checkpoint_name")
                and p.hr_checkpoint_name is not None
                and hasattr(p, "override_settings")
            ):
                with temp_dict(p.override_settings, sd_model_checkpoint=p.hr_checkpoint_name):
                    proc = process_images(p)
            else:
                proc = process_images(p)

        if output_images is not None:
            if not no_mask and p.image_mask is not None:
                entire_mask_image = numpy.array(p.image_mask)
                entire_image = numpy.array(p.init_images[0])
                add_image(output_images, MaskGenerator.to_masked_image(entire_mask_image, entire_image))

            output_image = proc.images[0]
            if strength > 0:
                output_image = add_comment(output_image, f"strength: {strength}", top=True)
            if len(pp) > 0 and p.prompt != pp:
                output_image = add_comment(output_image, pp)

            add_image(output_images, output_image)

        p.init_images = proc.images
