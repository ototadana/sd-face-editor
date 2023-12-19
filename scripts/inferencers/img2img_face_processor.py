from typing import Union

from modules.processing import StableDiffusionProcessingImg2Img, process_images
from PIL.Image import Image

from scripts.entities.face import Face
from scripts.entities.option import Option
from scripts.inferencers.temp_settings import temp_attr, temp_dict
from scripts.use_cases.face_processor import FaceProcessor


class Img2ImgFaceProcessor(FaceProcessor):
    def name(self) -> str:
        return "img2img"

    def process(
        self,
        face: Face,
        p: StableDiffusionProcessingImg2Img,
        strength1: Union[float, int],
        option: Option,
        pp: str = "",
        np: str = "",
        use_refiner_model_only=False,
        ignore_larger_faces=None,
        **kwargs,
    ) -> Image:
        face_size = option.face_size
        ignore_larger_faces = ignore_larger_faces if ignore_larger_faces is not None else option.ignore_larger_faces
        if ignore_larger_faces and face.width > face_size:
            face.info = f"ignore larger face:\n {face.width}x{face.height} > {face_size}x{face_size}"
            return face.image

        p.init_images = [face.image]
        p.width = face.image.width
        p.height = face.image.height
        p.denoising_strength = strength1
        p.do_not_save_samples = True

        print(f"prompt for the {face.face_area.tag}: {p.prompt}")
        with temp_attr(
            p,
            prompt=pp if len(pp) > 0 else p.prompt,
            negative_prompt=np if len(np) > 0 else p.negative_prompt,
            refiner_switch_at=0 if use_refiner_model_only else p.refiner_switch_at,
            overlay_images=[],
            image_mask=None,
            mask=None,
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

            p.init_images = proc.images
            return proc.images[0]
