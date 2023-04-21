from operator import attrgetter

import cv2
import gradio as gr
import modules.scripts as scripts
import modules.shared as shared
import numpy as np
import torch
from facexlib.detection import RetinaFace, init_detection_model, retinaface
from facexlib.parsing import BiSeNet, init_parsing_model
from facexlib.utils.misc import img2tensor
from modules.processing import (Processed, StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img,
                                process_images)
from PIL import Image
from torchvision.transforms.functional import normalize


class Face:
    def __init__(self, entire_image: np.ndarray, face_box: np.ndarray, face_margin: float):
        left, top, right, bottom = self.__to_square(face_box)

        self.left, self.top, self.right, self.bottom = self.__ensure_margin(
            left, top, right, bottom, entire_image, face_margin)

        self.width = self.right - self.left
        self.height = self.bottom - self.top

        self.image = self.__crop_face_image(entire_image)

    def __crop_face_image(self, entire_image: np.ndarray):
        cropped = entire_image[self.top: self.bottom, self.left: self.right, :]
        return Image.fromarray(
            cv2.resize(cropped, dsize=(512, 512)))

    def __to_square(self, face_box: np.ndarray):
        left, top, right, bottom, *_ = list(map(int, face_box))

        width = right - left
        height = bottom - top

        if width % 2 == 1:
            right = right + 1
            width = width + 1
        if height % 2 == 1:
            bottom = bottom + 1
            height = height + 1

        diff = int(abs(width - height) / 2)
        if width > height:
            top = top - diff
            bottom = bottom + diff
        else:
            left = left - diff
            right = right + diff

        return left, top, right, bottom

    def __ensure_margin(self, left: int, top: int, right: int, bottom: int, entire_image: np.ndarray, margin: float):
        entire_height, entire_width = entire_image.shape[:2]

        side_length = right - left
        margin = min(min(entire_height, entire_width) /
                     side_length, margin)
        diff = int((side_length * margin - side_length) / 2)

        top = top - diff
        bottom = bottom + diff
        left = left - diff
        right = right + diff

        if top < 0:
            bottom = bottom - top
            top = 0
        if left < 0:
            right = right - left
            left = 0

        if bottom > entire_height:
            top = top - (bottom - entire_height)
            bottom = entire_height
        if right > entire_width:
            left = left - (right - entire_width)
            right = entire_width

        return left, top, right, bottom


class Script(scripts.Script):
    def title(self):
        return "Face Editor"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        max_face_count = gr.Slider(
            minimum=1,
            maximum=20,
            step=1,
            value=20,
            label="Maximum number of faces to detect",
        )
        confidence = gr.Slider(
            minimum=0.7,
            maximum=1.0,
            step=0.01,
            value=0.97,
            label="Face detection confidence",
        )
        face_margin = gr.Slider(
            minimum=1.0, maximum=2.0, step=0.1, value=1.6, label="Face margin"
        )
        prompt_for_face = gr.Textbox(
            show_label=False,
            placeholder="Prompt for face",
            label="Prompt for face",
            lines=2,
        )
        strength1 = gr.Slider(
            minimum=0.1,
            maximum=0.8,
            step=0.05,
            value=0.4,
            label="Denoising strength for face images",
        )
        mask_size = gr.Slider(label="Mask size", minimum=0,
                              maximum=64, step=1, value=0)
        mask_blur = gr.Slider(label="Mask blur", minimum=0,
                              maximum=64, step=1, value=0)
        strength2 = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            value=0.15,
            label="Denoising strength for the entire image",
        )

        apply_inside_mask_only = gr.Checkbox(
            label="Apply inside mask only",
            value=False
        )

        save_original_image = gr.Checkbox(
            label="Save original image",
            value=False,
            visible=not is_img2img)

        show_intermediate_steps = gr.Checkbox(
            label="Show intermediate steps",
            value=False,
            visible=is_img2img)

        return [
            face_margin,
            confidence,
            strength1,
            strength2,
            max_face_count,
            mask_size,
            mask_blur,
            prompt_for_face,
            apply_inside_mask_only,
            save_original_image,
            show_intermediate_steps,
        ]

    def run(
        self,
        o: StableDiffusionProcessing,
        face_margin: float,
        confidence: float,
        strength1: float,
        strength2: float,
        max_face_count: int,
        mask_size: int,
        mask_blur: int,
        prompt_for_face: str,
        apply_inside_mask_only: bool,
        save_original_image: bool,
        show_intermediate_steps: bool,
    ):
        if hasattr(retinaface, 'device'):
            retinaface.device = shared.device

        mask_model = init_parsing_model(device=shared.device)
        detection_model = init_detection_model(
            "retinaface_resnet50", device=shared.device
        )

        if isinstance(o, StableDiffusionProcessingImg2Img):
            return self.__proc_image(o, mask_model, detection_model,
                                     face_margin=face_margin, confidence=confidence,
                                     strength1=strength1, strength2=strength2,
                                     max_face_count=max_face_count, mask_size=mask_size,
                                     mask_blur=mask_blur, prompt_for_face=prompt_for_face,
                                     apply_inside_mask_only=apply_inside_mask_only,
                                     show_intermediate_steps=show_intermediate_steps)
        else:
            shared.state.job_count = o.n_iter * 3
            if not save_original_image:
                o.do_not_save_samples = True
            res = process_images(o)
            o.do_not_save_samples = False

            edited_images = []
            seed_index = 0
            for i, image in enumerate(res.images):
                if i < res.index_of_first_image:
                    continue

                p = StableDiffusionProcessingImg2Img(init_images=[image])
                p.__dict__.update(o.__dict__)
                p.width, p.height = image.size
                if seed_index < len(res.all_seeds):
                    p.seed = res.all_seeds[seed_index]
                    seed_index += 1
                proc = self.__proc_image(p, mask_model, detection_model,
                                         face_margin=face_margin, confidence=confidence,
                                         strength1=strength1, strength2=strength2,
                                         max_face_count=max_face_count, mask_size=mask_size,
                                         mask_blur=mask_blur, prompt_for_face=prompt_for_face,
                                         apply_inside_mask_only=apply_inside_mask_only,
                                         pre_proc_image=image)
                edited_images.append(proc.images[-1])
            res.images.extend(edited_images)
            return res

    def __proc_image(self, p: StableDiffusionProcessingImg2Img,
                     mask_model: BiSeNet,
                     detection_model: RetinaFace,
                     face_margin: float,
                     confidence: float,
                     strength1: float,
                     strength2: float,
                     max_face_count: int,
                     mask_size: int,
                     mask_blur: int,
                     prompt_for_face: str,
                     apply_inside_mask_only: bool,
                     pre_proc_image: Image = None,
                     show_intermediate_steps: bool = False) -> Processed:
        entire_image = np.array(p.init_images[0])
        faces = self.__crop_face(
            detection_model, p.init_images[0], face_margin, confidence)
        faces = faces[:max_face_count]
        entire_mask_image = np.zeros_like(entire_image)

        entire_width = (p.width // 8) * 8
        entire_height = (p.height // 8) * 8
        entire_prompt = p.prompt
        p.batch_size = 1
        p.n_iter = 1
        scripts = p.scripts

        if shared.state.job_count == -1:
            shared.state.job_count = len(faces) + 1

        print(f"number of faces: {len(faces)}")
        if len(faces) == 0 and pre_proc_image is not None:
            return Processed(p, images_list=[pre_proc_image])
        output_images = []
        p.scripts = None

        for face in faces:
            if shared.state.interrupted:
                break

            p.init_images = [face.image]
            p.width = face.image.width
            p.height = face.image.height
            p.denoising_strength = strength1
            p.prompt = prompt_for_face if len(
                prompt_for_face) > 0 else entire_prompt
            p.do_not_save_samples = True

            proc = process_images(p)

            if proc.images[0].mode != 'RGB':
                proc.images[0] = proc.images[0].convert('RGB')

            face_image = np.array(proc.images[0])
            mask_image = self.__to_mask_image(
                mask_model, face_image, mask_size)

            if show_intermediate_steps:
                output_images.append(Image.fromarray(face_image))
                output_images.append(Image.fromarray(
                    self.__to_masked_image(mask_image, face_image)))

            face_image = cv2.resize(face_image, dsize=(
                face.width, face.height))
            mask_image = cv2.resize(mask_image, dsize=(
                face.width, face.height))

            if apply_inside_mask_only:
                face_background = entire_image[
                    face.top: face.bottom,
                    face.left: face.right,
                ]
                face_fg = (face_image * (mask_image/255.0)).astype('uint8')
                face_bg = (face_background *
                           (1 - (mask_image/255.0))).astype('uint8')
                face_image = face_fg + face_bg

            entire_image[
                face.top: face.bottom,
                face.left: face.right,
            ] = face_image
            entire_mask_image[
                face.top: face.bottom,
                face.left: face.right,
            ] = mask_image

        p.scripts = scripts
        p.prompt = entire_prompt
        p.width = entire_width
        p.height = entire_height
        p.init_images = [Image.fromarray(entire_image)]
        p.denoising_strength = strength2
        p.mask_blur = mask_blur
        p.inpainting_mask_invert = 1
        p.inpainting_fill = 1
        p.image_mask = Image.fromarray(entire_mask_image)
        p.do_not_save_samples = False
        proc = process_images(p)

        if show_intermediate_steps:
            output_images.append(p.init_images[0])
            output_images.append(Image.fromarray(
                self.__to_masked_image(entire_mask_image, entire_image)))
            output_images.append(proc.images[0])
            proc.images = output_images

        return proc

    def __to_masked_image(self, mask_image: np.ndarray, image: np.ndarray) -> np.ndarray:
        gray_mask = np.where(mask_image == 0, 47, 255) / 255.0
        return (image * gray_mask).astype('uint8')

    def __crop_face(self, detection_model: RetinaFace, image: Image, face_margin: float, confidence: float) -> list:
        with torch.no_grad():
            face_boxes, _ = detection_model.align_multi(image, confidence)
            return self.__crop(image, face_boxes, face_margin)

    def __crop(self, image: Image, face_boxes: list, face_margin: float) -> list:
        image = np.array(image, dtype=np.uint8)

        areas = []
        for face_box in face_boxes:
            areas.append(Face(image, face_box, face_margin))

        return sorted(areas, key=attrgetter("height"), reverse=True)

    def __to_mask_image(self, mask_model: BiSeNet, face_image: Image, mask_size: int) -> np.ndarray:
        face_image = np.array(face_image)
        face_tensor = img2tensor(face_image.astype(
            "float32") / 255.0, float32=True)
        normalize(face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        face_tensor = torch.unsqueeze(face_tensor, 0).to(shared.device)

        with torch.no_grad():
            face = mask_model(face_tensor)[0]
        face = face.squeeze(0).cpu().numpy().argmax(0)
        face = face.copy().astype(np.uint8)

        mask = self.__to_mask(face)
        if mask_size > 0:
            mask = cv2.dilate(mask, np.empty(
                0, np.uint8), iterations=mask_size)
        return mask

    def __to_mask(self, face: np.ndarray) -> np.ndarray:
        mask = np.zeros((face.shape[0], face.shape[1], 3), dtype=np.uint8)
        num_of_class = np.max(face)
        for i in range(1, num_of_class + 1):
            index = np.where(face == i)
            if i < 14:
                mask[index[0], index[1], :] = [255, 255, 255]
        return mask
