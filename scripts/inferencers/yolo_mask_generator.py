from typing import Tuple

import cv2
import modules.shared as shared
import numpy as np

from scripts.inferencers.yolo_inferencer import YoloInferencer
from scripts.use_cases.mask_generator import MaskGenerator


class YoloMaskGenerator(MaskGenerator, YoloInferencer):
    def __init__(self):
        super().__init__("segment")

    def name(self) -> str:
        return "YOLO"

    def generate_mask(
        self,
        face_image: np.ndarray,
        face_area_on_image: Tuple[int, int, int, int],
        path: str = "yolov8n-seg.pt",
        repo_id: str = None,
        filename: str = None,
        conf: float = 0.5,
        **kwargs,
    ) -> np.ndarray:
        self.load_model(path, repo_id, filename)

        output = self.model.predict(face_image, device=shared.device)

        combined_mask = np.zeros(face_image.shape[:2], np.uint8)

        for detection in output:
            boxes = detection.boxes
            for i, box in enumerate(boxes):
                box_conf = float(box.conf[0])
                if box_conf < conf:
                    continue
                mask = cv2.resize(detection.masks[i].data[0].cpu().numpy(), (512, 512))
                combined_mask += (mask * 255).astype(np.uint8)

        return np.dstack([combined_mask] * 3)
