from typing import Any, Dict

from scripts.use_cases.face_detector import FaceDetector
from scripts.use_cases.mask_generator import MaskGenerator


class InferencerSet:
    def __init__(
        self,
        face_detector: FaceDetector,
        face_detector_params: Dict[str, Any],
        mask_generator: MaskGenerator,
        mask_generator_params: Dict[str, Any],
    ):
        self.mask_generator = mask_generator
        self.mask_generator_params = mask_generator_params
        self.face_detector = face_detector
        self.face_detector_params = face_detector_params
