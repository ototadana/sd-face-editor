from typing import Union

from PIL import Image

from modules.processing import StableDiffusionProcessingImg2Img, process_images
from scripts.entities.face import Face
from scripts.use_cases.face_processor import FaceProcessor


class Img2ImgFaceProcessor(FaceProcessor):
    def name(self) -> str:
        return "img2img"

    def process(
        self,
        face: Face,
        p: StableDiffusionProcessingImg2Img,
        strength1: Union[float, int],
        pp: str = "",
        np: str = "",
        use_refiner_model_only=False,
        **kwargs,
    ) -> Image:
        p.init_images = [face.image]
        p.width = face.image.width
        p.height = face.image.height
        p.denoising_strength = strength1
        p.do_not_save_samples = True

        if len(pp) > 0:
            p.prompt = pp
        if len(np) > 0:
            p.negative_prompt = np

        if use_refiner_model_only:
            refiner_switch_at = p.refiner_switch_at
            p.refiner_switch_at = 0

        has_hr_checkpoint_name = p.enable_hr and hasattr(p, "hr_checkpoint_name") and p.hr_checkpoint_name is not None and hasattr(p, "override_settings")
        if has_hr_checkpoint_name:
            backup_sd_model_checkpoint = p.override_settings.get("sd_model_checkpoint", None)
            p.override_settings["sd_model_checkpoint"] = p.hr_checkpoint_name    

        print(f"prompt for the {face.face_area.tag}: {p.prompt}")

        proc = process_images(p)
        if use_refiner_model_only:
            p.refiner_switch_at = refiner_switch_at
        if has_hr_checkpoint_name:
            p.override_settings["sd_model_checkpoint"] = backup_sd_model_checkpoint

        return proc.images[0]
