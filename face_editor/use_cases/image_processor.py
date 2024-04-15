import os
import tempfile
from operator import attrgetter
from typing import Any, List, Tuple

import cv2
import modules.images as images
import modules.scripts as scripts
import modules.shared as shared
import numpy as np
from face_editor.entities.definitions import Rule
from face_editor.entities.face import Face
from face_editor.entities.option import Option
from face_editor.entities.rect import Rect
from face_editor.use_cases.mask_generator import MaskGenerator
from face_editor.use_cases.registry import face_processors
from face_editor.use_cases.workflow_manager import WorkflowManager
from modules.processing import (
    Processed,
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
    create_infotext,
    process_images,
)
from PIL import Image

os.makedirs(os.path.join(tempfile.gettempdir(), "gradio"), exist_ok=True)


class ImageProcessor:
    def __init__(self, workflow: WorkflowManager) -> None:
        self.workflow = workflow

    def proc_images(self, o: StableDiffusionProcessing, res: Processed, option: Option):
        edited_images, all_seeds, all_prompts, all_negative_prompts, infotexts = [], [], [], [], []
        seed_index = 0
        subseed_index = 0

        self.__extend_infos(res.all_prompts, len(res.images))
        self.__extend_infos(res.all_negative_prompts, len(res.images))
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

            proc = self.proc_image(p, option, image, res.infotexts[i], (res.width, res.height))
            edited_images.extend(proc.images)
            all_seeds.extend(proc.all_seeds)
            all_prompts.extend(proc.all_prompts)
            all_negative_prompts.extend(proc.all_negative_prompts)
            infotexts.extend(proc.infotexts)

        if res.index_of_first_image == 1:
            edited_images.insert(0, images.image_grid(edited_images))
            infotexts.insert(0, infotexts[0])

        if option.show_original_image:
            res.images.extend(edited_images)
            res.infotexts.extend(infotexts)
            res.all_seeds.extend(all_seeds)
            res.all_prompts.extend(all_prompts)
            res.all_negative_prompts.extend(all_negative_prompts)
        else:
            res.images = edited_images
            res.infotexts = infotexts
            res.all_seeds = all_seeds
            res.all_prompts = all_prompts
            res.all_negative_prompts = all_negative_prompts

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
        self,
        p: StableDiffusionProcessingImg2Img,
        option: Option,
        pre_proc_image: Image = None,
        pre_proc_infotext: Any = None,
        original_size: Tuple[int, int] = None,
    ) -> Processed:
        if shared.opts.data.get("face_editor_auto_face_size_by_model", False):
            option.face_size = 1024 if getattr(shared.sd_model, "is_sdxl", False) else 512
        params = option.to_dict()

        if hasattr(p.init_images[0], "mode") and p.init_images[0].mode != "RGB":
            p.init_images[0] = p.init_images[0].convert("RGB")

        entire_image = np.array(p.init_images[0])
        faces = self.__crop_face(p.init_images[0], option)
        faces = faces[: option.max_face_count]
        faces = sorted(faces, key=attrgetter("center"))
        entire_mask_image = np.zeros_like(entire_image)

        entire_width = (p.width // 8) * 8
        entire_height = (p.height // 8) * 8
        entire_prompt = p.prompt
        entire_all_prompts = p.all_prompts
        original_denoising_strength = p.denoising_strength
        p.batch_size = 1
        p.n_iter = 1

        if shared.state.job_count == -1:
            shared.state.job_count = len(faces) + 1

        output_images = []
        if option.show_intermediate_steps:
            output_images.append(self.__show_detected_faces(np.copy(entire_image), faces, p))

        print(f"number of faces: {len(faces)}.  ")
        if (
            len(faces) == 0
            and pre_proc_image is not None
            and (
                option.save_original_image
                or not shared.opts.data.get("face_editor_save_original_on_detection_fail", False)
            )
        ):
            return Processed(
                p,
                images_list=[pre_proc_image],
                all_prompts=[p.prompt],
                all_seeds=[p.seed],
                infotexts=[pre_proc_infotext],
            )

        wildcards_script = self.__get_wildcards_script(p)
        face_prompts = self.__get_face_prompts(len(faces), option.prompt_for_face, entire_prompt)

        if not option.apply_scripts_to_faces:
            p.scripts = None

        for i, face in enumerate(faces):
            if shared.state.interrupted:
                break

            p.prompt = face_prompts[i]
            if wildcards_script is not None:
                p.prompt = self.__apply_wildcards(wildcards_script, p.prompt, i)

            rule = self.workflow.select_rule(faces, i, entire_width, entire_height)

            if rule is None or len(rule.then) == 0:
                continue
            jobs = rule.then

            if option.show_intermediate_steps:
                original_face = np.array(face.image.copy())

            proc_image = self.workflow.process(jobs, face, p, option)
            if proc_image is None:
                continue

            if proc_image.mode != "RGB":
                proc_image = proc_image.convert("RGB")

            face_image = np.array(proc_image)
            mask_image = self.workflow.generate_mask(jobs, face_image, face, option)

            if option.show_intermediate_steps:
                feature = self.__get_feature(p.prompt, entire_prompt)
                self.__add_debug_image(
                    output_images, face, original_face, face_image, mask_image, rule, option, feature
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
        simple_composite_image = Image.fromarray(entire_image)
        p.init_images = [simple_composite_image]
        p.mask_blur = option.mask_blur
        p.inpainting_mask_invert = 1
        p.inpainting_fill = 1
        p.image_mask = Image.fromarray(entire_mask_image)
        p.do_not_save_samples = True

        p.extra_generation_params.update(params)

        if option.strength2 > 0:
            p.denoising_strength = option.strength2
            if p.scripts is None:
                p.scripts = scripts.ScriptRunner()
            proc = process_images(p)
            p.init_images = proc.images

        if original_size is not None:
            p.width, p.height = original_size
        p.denoising_strength = original_denoising_strength
        proc = self.__save_images(p, pre_proc_infotext if len(faces) == 0 else None)

        if option.show_intermediate_steps:
            output_images.append(simple_composite_image)
            if option.strength2 > 0:
                output_images.append(Image.fromarray(self.__to_masked_image(entire_mask_image, entire_image)))
                output_images.append(proc.images[0])
            proc.images = output_images

        self.__extend_infos(proc.all_prompts, len(proc.images))
        self.__extend_infos(proc.all_seeds, len(proc.images))
        self.__extend_infos(proc.infotexts, len(proc.images))

        return proc

    def __add_debug_image(
        self,
        output_images: list,
        face: Face,
        original_face: np.ndarray,
        face_image: np.ndarray,
        mask_image: np.ndarray,
        rule: Rule,
        option: Option,
        feature: str,
    ):
        h, w = original_face.shape[:2]
        debug_image = np.zeros_like(original_face)

        tag = f"{face.face_area.tag} ({face.face_area.width}x{face.face_area.height})"
        attributes = str(face.face_area.attributes) if face.face_area.attributes else ""
        if not attributes:
            attributes = option.upscaler if option.upscaler != Option.DEFAULT_UPSCALER else ""

        def resize(img: np.ndarray):
            return cv2.resize(img, (w // 2, h // 2))

        debug_image[0 : h // 2, 0 : w // 2] = resize(
            self.__add_comment(self.__add_comment(original_face, attributes), tag, True)
        )

        criteria = rule.when.criteria if rule.when is not None and rule.when.criteria is not None else ""
        debug_image[0 : h // 2, w // 2 :] = resize(
            self.__add_comment(self.__add_comment(face_image, feature), criteria, True)
        )

        coverage = MaskGenerator.calculate_mask_coverage(mask_image) * 100
        mask_info = f"size:{option.mask_size}, blur:{option.mask_blur}, cov:{coverage:.0f}%"
        debug_image[h // 2 :, 0 : w // 2] = resize(
            self.__add_comment(self.__to_masked_image(mask_image, face_image), mask_info)
        )

        face_fg = (face_image * (mask_image / 255.0)).astype("uint8")
        face_bg = (original_face * (1 - (mask_image / 255.0))).astype("uint8")
        debug_image[h // 2 :, w // 2 :] = resize(face_fg + face_bg)

        output_images.append(Image.fromarray(debug_image))

    def __show_detected_faces(self, entire_image: np.ndarray, faces: List[Face], p: StableDiffusionProcessingImg2Img):
        processor = face_processors.get("debug")
        for face in faces:
            face_image = np.array(processor.process(face, p))
            face_image = cv2.resize(face_image, dsize=(face.width, face.height), interpolation=cv2.INTER_AREA)
            entire_image[
                face.top : face.bottom,
                face.left : face.right,
            ] = face_image
        return Image.fromarray(self.__add_comment(entire_image, f"{len(faces)}"))

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

    def __add_comment(self, image: np.ndarray, comment: str, top: bool = False) -> np.ndarray:
        image = np.copy(image)
        h, _, _ = image.shape
        pos = (10, 48) if top else (10, h - 16)
        cv2.putText(
            image,
            text=comment,
            org=pos,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.2,
            color=(0, 0, 0),
            thickness=10,
        )
        cv2.putText(
            image,
            text=comment,
            org=pos,
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

    def __save_images(self, p: StableDiffusionProcessingImg2Img, pre_proc_infotext: Any) -> Processed:
        if pre_proc_infotext is not None:
            infotext = pre_proc_infotext
        else:
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
        gray_mask = mask_image / 255.0
        return (image * gray_mask).astype("uint8")

    def __crop_face(self, image: Image, option: Option) -> List[Face]:
        face_areas = self.workflow.detect_faces(image, option)
        return self.__crop(
            image, face_areas, option.face_margin, option.face_size, option.ignore_larger_faces, option.upscaler
        )

    def __crop(
        self,
        image: Image,
        face_areas: List[Rect],
        face_margin: float,
        face_size: int,
        ignore_larger_faces: bool,
        upscaler: str,
    ) -> List[Face]:
        image = np.array(image, dtype=np.uint8)

        areas: List[Face] = []
        for face_area in face_areas:
            face = Face(image, face_area, face_margin, face_size, upscaler)
            if ignore_larger_faces and face.width > face_size:
                continue
            areas.append(face)

        return sorted(areas, key=attrgetter("height"), reverse=True)
