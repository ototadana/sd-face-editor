from typing import List

import cv2
import modules.shared as shared
import numpy as np
import torch
from facexlib.parsing import init_parsing_model
from facexlib.utils.misc import img2tensor
from PIL import Image
from torchvision.transforms.functional import normalize

from scripts.use_cases.mask_generator import MaskGenerator


class BiSeNetMaskGenerator(MaskGenerator):
    def __init__(self) -> None:
        self.mask_model = init_parsing_model(device=shared.device)

    def generate_mask(self, face_image: Image, mask_size: int, targets: List[str]) -> np.ndarray:
        face_image = np.array(face_image)
        h, w, _ = face_image.shape

        if w != 512 or h != 512:
            rw = (int(w * (512 / w)) // 8) * 8
            rh = (int(h * (512 / h)) // 8) * 8
            face_image = cv2.resize(face_image, dsize=(rw, rh))

        face_tensor = img2tensor(face_image.astype(
            "float32") / 255.0, float32=True)
        normalize(face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        face_tensor = torch.unsqueeze(face_tensor, 0).to(shared.device)

        with torch.no_grad():
            face = self.mask_model(face_tensor)[0]
        face = face.squeeze(0).cpu().numpy().argmax(0)
        face = face.copy().astype(np.uint8)

        mask = self.__to_mask(face, targets)
        if mask_size > 0:
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=mask_size)

        if w != 512 or h != 512:
            mask = cv2.resize(mask, dsize=(w, h))

        return mask

    def __to_mask(self, face: np.ndarray, targets: List[str]) -> np.ndarray:
        keep_face = "Face" in targets
        keep_neck = "Neck" in targets
        keep_hair = "Hair" in targets
        keep_hat = "Hat" in targets

        mask = np.zeros((face.shape[0], face.shape[1], 3), dtype=np.uint8)
        num_of_class = np.max(face)
        for i in range(1, num_of_class + 1):
            index = np.where(face == i)
            if i < 14 and keep_face:
                mask[index[0], index[1], :] = [255, 255, 255]
            elif i == 14 and keep_neck:
                mask[index[0], index[1], :] = [255, 255, 255]
            elif i == 17 and keep_hair:
                mask[index[0], index[1], :] = [255, 255, 255]
            elif i == 18 and keep_hat:
                mask[index[0], index[1], :] = [255, 255, 255]
        return mask
