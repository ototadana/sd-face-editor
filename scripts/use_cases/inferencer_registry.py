from scripts.use_cases.face_detector import FaceDetector
from scripts.use_cases.mask_generator import MaskGenerator


class InferencerRegistry:
    def __init__(self, face_detector: FaceDetector, mask_generator: MaskGenerator):
        self.mask_generator = mask_generator
        self.face_detector = face_detector
