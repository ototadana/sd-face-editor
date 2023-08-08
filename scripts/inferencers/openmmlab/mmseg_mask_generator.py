from typing import Tuple

import cv2
import modules.shared as shared
import numpy as np
from huggingface_hub import hf_hub_download
from mmseg.apis import inference_model, init_model

from scripts.use_cases.mask_generator import MaskGenerator


class MMSegMaskGenerator(MaskGenerator):
    def __init__(self):
        checkpoint_file = hf_hub_download(
            repo_id="ototadana/occlusion-aware-face-segmentation",
            filename="deeplabv3plus_r101_512x512_face-occlusion-93ec6695.pth",
        )
        config_file = hf_hub_download(
            repo_id="ototadana/occlusion-aware-face-segmentation",
            filename="deeplabv3plus_r101_512x512_face-occlusion.py",
        )
        self.model = init_model(config_file, checkpoint_file, device=shared.device)

    def name(self) -> str:
        return "MMSeg"

    def generate_mask(
        self,
        face_image: np.ndarray,
        face_area_on_image: Tuple[int, int, int, int],
        mask_size: int,
        use_minimal_area: bool,
        **kwargs,
    ) -> np.ndarray:
        face_image = face_image.copy()
        face_image = face_image[:, :, ::-1]

        if use_minimal_area:
            face_image = MaskGenerator.mask_non_face_areas(face_image, face_area_on_image)

        result = inference_model(self.model, face_image)
        pred_sem_seg = result.pred_sem_seg
        pred_sem_seg_data = pred_sem_seg.data.squeeze(0)
        pred_sem_seg_np = pred_sem_seg_data.cpu().numpy()
        pred_sem_seg_np = (pred_sem_seg_np * 255).astype(np.uint8)

        mask = cv2.cvtColor(pred_sem_seg_np, cv2.COLOR_BGR2RGB)
        if mask_size > 0:
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=mask_size)

        return mask
