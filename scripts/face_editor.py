from operator import attrgetter

import cv2
import gradio as gr
import modules.scripts as scripts
import modules.shared as shared
import numpy as np
import torch
from facexlib.detection import RetinaFace, init_detection_model
from facexlib.parsing import BiSeNet, init_parsing_model
from facexlib.utils.misc import img2tensor
from modules.processing import StableDiffusionProcessingImg2Img, process_images
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
        return is_img2img

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
        return [
            face_margin,
            confidence,
            strength1,
            strength2,
            max_face_count,
            mask_size,
            mask_blur,
            prompt_for_face,
        ]

    def run(
        self,
        p: StableDiffusionProcessingImg2Img,
        face_margin: float,
        confidence: float,
        strength1: float,
        strength2: float,
        max_face_count: int,
        mask_size: int,
        mask_blur: int,
        prompt_for_face: str,
    ):
        mask_model = init_parsing_model(device=shared.device)
        detection_model = init_detection_model(
            "retinaface_resnet50", device=shared.device
        )

        entire_image = np.array(p.init_images[0])
        faces = self.__crop_face(
            detection_model, p.init_images[0], face_margin, confidence)
        faces = faces[:max_face_count]
        entire_mask_image = np.zeros_like(entire_image)

        entire_width = p.width
        entire_height = p.height
        entire_prompt = p.prompt
        p.batch_size = 1

        shared.state.job_count = len(faces) + 1

        print(f"number of faces: {len(faces)}")
        output_images = []

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

            face_image = np.array(proc.images[0])
            mask_image = self.__to_mask_image(
                mask_model, face_image, mask_size)

            output_images.append(face_image)
            output_images.append(
                self.__to_masked_image(mask_image, face_image))

            face_image = cv2.resize(face_image, dsize=(
                face.width, face.height))
            mask_image = cv2.resize(mask_image, dsize=(
                face.width, face.height))

            entire_image[
                face.top: face.bottom,
                face.left: face.right,
            ] = face_image
            entire_mask_image[
                face.top: face.bottom,
                face.left: face.right,
            ] = mask_image

        output_images.append(entire_image)
        output_images.append(self.__to_masked_image(
            entire_mask_image, entire_image))

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

        final_image = proc.images[0]
        proc.images = output_images
        proc.images.append(final_image)

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
        face_tensor = torch.unsqueeze(face_tensor, 0).cuda()

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
