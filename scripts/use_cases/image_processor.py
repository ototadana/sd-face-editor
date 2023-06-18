import os
import tempfile
from operator import attrgetter
from typing import List

import cv2
import modules.images as images
import modules.scripts as scripts
import modules.shared as shared
import numpy as np
from modules.processing import (
    Processed,
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
    create_infotext,
    process_images,
)
from PIL import Image

from scripts.entities.face import Face
from scripts.entities.option import Option
from scripts.entities.rect import Rect
from scripts.use_cases.inferencer_set import InferencerSet

os.makedirs(os.path.join(tempfile.gettempdir(), "gradio"), exist_ok=True)


class ImageProcessor:
    def __init__(self, inferencers: InferencerSet) -> None:
        self.face_detector = inferencers.face_detector
        self.face_detector_params = inferencers.face_detector_params
        self.mask_generator = inferencers.mask_generator
        self.mask_generator_params = inferencers.mask_generator_params

    def proc_images(self, o: StableDiffusionProcessing, res: Processed, option: Option):
        edited_images, all_seeds, all_prompts, infotexts = [], [], [], []
        seed_index = 0
        subseed_index = 0

        self.__extend_infos(res.all_prompts, len(res.images))
        self.__extend_infos(res.all_seeds, len(res.images))
        self.__extend_infos(res.infotexts, len(res.images))

        for i, image in enumerate(res.images):
            if i < res.index_of_first_image:
                continue

            p = StableDiffusionProcessingImg2Img(init_images=[image])
            self.__init_processing(p, o, image)

            if seed_index < len(res.all_seeds):
                p.seed = res.all_seeds[seed_index]
                seed_index += 1
            if subseed_index < len(res.all_subseeds):
                p.subseed = res.all_subseeds[subseed_index]
                subseed_index += 1

            if type(p.prompt) == list:
                p.prompt = p.prompt[i]

            proc = self.proc_image(p, option, image)
            edited_images.extend(proc.images)
            all_seeds.extend(proc.all_seeds)
            all_prompts.extend(proc.all_prompts)
            infotexts.extend(proc.infotexts)

        if option.show_original_image:
            res.images.extend(edited_images)
        else:
            res.images = edited_images
        res.all_seeds.extend(all_seeds)
        res.all_prompts.extend(all_prompts)
        res.infotexts.extend(infotexts)
        return res

    def __init_processing(self, p: StableDiffusionProcessingImg2Img, o: StableDiffusionProcessing, image):
        sample = p.sample
        p.__dict__.update(o.__dict__)
        p.sampler = None
        p.c = None
        p.uc = None
        p.cached_c = [None, None]
        p.cached_uc = [None, None]
        p.init_images = [image]
        p.width, p.height = image.size
        p.sample = sample

    def proc_image(
        self, p: StableDiffusionProcessingImg2Img, option: Option, pre_proc_image: Image = None
    ) -> Processed:
        params = option.to_dict()

        if hasattr(p.init_images[0], "mode") and p.init_images[0].mode != "RGB":
            p.init_images[0] = p.init_images[0].convert("RGB")

        entire_image = np.array(p.init_images[0])
        faces = self.__crop_face(
            p.init_images[0], option.face_margin, option.confidence, option.face_size, option.ignore_larger_faces
        )
        faces = faces[: option.max_face_count]
        faces = sorted(faces, key=attrgetter("center"))
        entire_mask_image = np.zeros_like(entire_image)

        entire_width = (p.width // 8) * 8
        entire_height = (p.height // 8) * 8
        entire_prompt = p.prompt
        entire_all_prompts = p.all_prompts
        p.batch_size = 1
        p.n_iter = 1

        if shared.state.job_count == -1:
            shared.state.job_count = len(faces) + 1

        print(f"number of faces: {len(faces)}")
        if len(faces) == 0 and pre_proc_image is not None:
            return Processed(
                p, images_list=[pre_proc_image], all_prompts=[p.prompt], all_seeds=[p.seed], infotexts=[""]
            )
        output_images = []

        wildcards_script = self.__get_wildcards_script(p)
        face_prompts = self.__get_face_prompts(len(faces), option.prompt_for_face, entire_prompt)
        face_prompt_index = 0

        if not option.apply_scripts_to_faces:
            p.scripts = None

        for face in faces:
            if shared.state.interrupted:
                break

            p.init_images = [face.image]
            p.width = face.image.width
            p.height = face.image.height
            p.denoising_strength = option.strength1
            p.prompt = face_prompts[face_prompt_index]
            if wildcards_script is not None:
                p.prompt = self.__apply_wildcards(wildcards_script, p.prompt, face_prompt_index)
            face_prompt_index += 1
            print(f"prompt for the face: {p.prompt}")

            p.do_not_save_samples = True

            proc = process_images(p)

            if proc.images[0].mode != "RGB":
                proc.images[0] = proc.images[0].convert("RGB")

            face_image = np.array(proc.images[0])
            self.mask_generator_params["mask_size"] = option.mask_size
            self.mask_generator_params["use_minimal_area"] = option.use_minimal_area
            self.mask_generator_params["affected_areas"] = option.affected_areas
            mask_image = self.mask_generator.generate_mask(
                face_image, face.face_area_on_image, **self.mask_generator_params
            )

            if option.mask_blur > 0:
                mask_image = cv2.blur(mask_image, (option.mask_blur, option.mask_blur))

            if option.show_intermediate_steps:
                feature = self.__get_feature(p.prompt, entire_prompt)
                mask_info = f"size:{option.mask_size}, blur:{option.mask_blur}"
                output_images.append(Image.fromarray(self.__add_comment(face_image, feature)))
                output_images.append(
                    Image.fromarray(self.__add_comment(self.__to_masked_image(mask_image, face_image), mask_info))
                )

            face_image = cv2.resize(face_image, dsize=(face.width, face.height), interpolation=cv2.INTER_AREA)
            mask_image = cv2.resize(mask_image, dsize=(face.width, face.height), interpolation=cv2.INTER_AREA)

            if option.use_minimal_area:
                l, t, r, b = face.face_area.to_tuple()
                face_image = face_image[t - face.top : b - face.top, l - face.left : r - face.left]
                mask_image = mask_image[t - face.top : b - face.top, l - face.left : r - face.left]
                face.top = t
                face.left = l
                face.bottom = b
                face.right = r

            if option.apply_inside_mask_only:
                face_background = entire_image[
                    face.top : face.bottom,
                    face.left : face.right,
                ]
                face_fg = (face_image * (mask_image / 255.0)).astype("uint8")
                face_bg = (face_background * (1 - (mask_image / 255.0))).astype("uint8")
                face_image = face_fg + face_bg

            entire_image[
                face.top : face.bottom,
                face.left : face.right,
            ] = face_image
            entire_mask_image[
                face.top : face.bottom,
                face.left : face.right,
            ] = mask_image

        p.prompt = entire_prompt
        p.all_prompts = entire_all_prompts
        p.width = entire_width
        p.height = entire_height
        p.init_images = [Image.fromarray(entire_image)]
        p.denoising_strength = option.strength2
        p.mask_blur = option.mask_blur
        p.inpainting_mask_invert = 1
        p.inpainting_fill = 1
        p.image_mask = Image.fromarray(entire_mask_image)
        p.do_not_save_samples = False

        p.extra_generation_params.update(params)

        if p.denoising_strength > 0:
            proc = process_images(p)
        else:
            proc = self.__save_images(p)

        if option.show_intermediate_steps:
            output_images.append(p.init_images[0])
            if p.denoising_strength > 0:
                output_images.append(Image.fromarray(self.__to_masked_image(entire_mask_image, entire_image)))
                output_images.append(proc.images[0])
            proc.images = output_images

        self.__extend_infos(proc.all_prompts, len(proc.images))
        self.__extend_infos(proc.all_seeds, len(proc.images))
        self.__extend_infos(proc.infotexts, len(proc.images))

        return proc

    def __get_wildcards_script(self, p: StableDiffusionProcessingImg2Img):
        if p.scripts is None:
            return None

        for script in p.scripts.alwayson_scripts:
            if script.filename.endswith("stable-diffusion-webui-wildcards/scripts/wildcards.py"):
                return script

        return None

    def __get_feature(self, prompt: str, entire_prompt: str) -> str:
        if prompt == "" or prompt == entire_prompt:
            return ""
        return prompt.replace(entire_prompt, "")

    def __add_comment(self, image: np.ndarray, comment: str) -> np.ndarray:
        image = np.copy(image)
        h, _, _ = image.shape
        cv2.putText(
            image,
            text=comment,
            org=(10, h - 16),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.2,
            color=(0, 0, 0),
            thickness=10,
        )
        cv2.putText(
            image,
            text=comment,
            org=(10, h - 16),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.2,
            color=(255, 255, 255),
            thickness=2,
        )
        return image

    def __apply_wildcards(self, wildcards_script: scripts.Script, prompt: str, seed: int) -> str:
        if "__" in prompt:
            wp = StableDiffusionProcessing()
            wp.all_prompts = [prompt]
            wp.all_seeds = [0 if shared.opts.wildcards_same_seed else seed]
            wildcards_script.process(wp)
            return wp.all_prompts[0]
        return prompt

    def __get_face_prompts(self, length: int, prompt_for_face: str, entire_prompt: str) -> List[str]:
        if len(prompt_for_face) == 0:
            return [entire_prompt] * length
        prompts = []
        p = prompt_for_face.split("||")
        for i in range(length):
            if i >= len(p):
                i = 0
            prompts.append(self.__edit_face_prompt(p[i], p[0], entire_prompt))
        return prompts

    def __edit_face_prompt(self, prompt: str, default_prompt: str, entire_prompt: str) -> str:
        if len(prompt) == 0:
            return default_prompt

        return prompt.strip().replace("@@", entire_prompt)

    def __save_images(self, p: StableDiffusionProcessingImg2Img) -> Processed:
        if p.all_prompts is None or len(p.all_prompts) == 0:
            p.all_prompts = [p.prompt]
        if p.all_negative_prompts is None or len(p.all_negative_prompts) == 0:
            p.all_negative_prompts = [p.negative_prompt]
        if p.all_seeds is None or len(p.all_seeds) == 0:
            p.all_seeds = [p.seed]
        if p.all_subseeds is None or len(p.all_subseeds) == 0:
            p.all_subseeds = [p.subseed]
        infotext = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, {}, 0, 0)
        images.save_image(
            p.init_images[0], p.outpath_samples, "", p.seed, p.prompt, shared.opts.samples_format, info=infotext, p=p
        )
        return Processed(
            p,
            images_list=p.init_images,
            seed=p.seed,
            info=infotext,
            subseed=p.subseed,
            index_of_first_image=0,
            infotexts=[infotext],
        )

    def __extend_infos(self, infos: list, image_count: int) -> None:
        infos.extend([infos[0]] * (image_count - len(infos)))

    def __to_masked_image(self, mask_image: np.ndarray, image: np.ndarray) -> np.ndarray:
        gray_mask = np.where(mask_image == 0, 47, 255) / 255.0
        return (image * gray_mask).astype("uint8")

    def __crop_face(
        self, image: Image, face_margin: float, confidence: float, face_size: int, ignore_larger_faces: bool
    ) -> List[Face]:
        self.face_detector_params["confidence"] = confidence
        face_areas = self.face_detector.detect_faces(image, **self.face_detector_params)
        return self.__crop(image, face_areas, face_margin, face_size, ignore_larger_faces)

    def __crop(
        self, image: Image, face_areas: List[Rect], face_margin: float, face_size: int, ignore_larger_faces: bool
    ) -> List[Face]:
        image = np.array(image, dtype=np.uint8)

        areas: List[Face] = []
        for face_area in face_areas:
            face = Face(image, face_area, face_margin, face_size)
            if ignore_larger_faces and face.width > face_size:
                continue
            areas.append(face)

        return sorted(areas, key=attrgetter("height"), reverse=True)
