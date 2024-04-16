from typing import Tuple

import cv2
import modules.shared as shared
import numpy as np
from face_editor.inferencers.yolo.inferencer import YoloInferencer
from face_editor.use_cases.mask_generator import MaskGenerator


class YoloMaskGenerator(MaskGenerator, YoloInferencer):
    def __init__(self):
        super().__init__("segment")

    def name(self) -> str:
        return "YOLO"

    def generate_mask(
        self,
        face_image: np.ndarray,
        face_area_on_image: Tuple[int, int, int, int],
        use_minimal_area: bool,
        tag: str = "face",
        path: str = "yolov8n-seg.pt",
        repo_id: str = None,
        filename: str = None,
        conf: float = 0.5,
        **kwargs,
    ) -> np.ndarray:
        self.load_model(path, repo_id, filename)

        names = self.model.names
        output = self.model.predict(face_image, device=shared.device)

        if use_minimal_area:
            face_image = MaskGenerator.mask_non_face_areas(face_image, face_area_on_image)

        combined_mask = np.zeros(face_image.shape[:2], np.uint8)

        for detection in output:
            boxes = detection.boxes
            for i, box in enumerate(boxes):
                box_tag = names[int(box.cls)]
                box_conf = float(box.conf[0])
                if box_tag != tag or box_conf < conf:
                    continue
                if detection.masks is None:
                    print(f"This model may not support masks: {self.loaded_path}")
                    continue
                mask = cv2.resize(detection.masks[i].data[0].cpu().numpy(), (512, 512))
                combined_mask += (mask * 255).astype(np.uint8)

        return np.dstack([combined_mask] * 3)
