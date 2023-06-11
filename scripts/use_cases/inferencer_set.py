from enum import Enum
from typing import List

from scripts.use_cases.face_detector import FaceDetector
from scripts.use_cases.mask_generator import MaskGenerator


class InferencerSet:
    class Name(Enum):
        STANDARD = "standard"
        ANIME = "anime"

    @classmethod
    def names(cls) -> List[str]:
        return [member.value for member in cls.Name]

    def __init__(self, name: Name, face_detector: FaceDetector, mask_generator: MaskGenerator):
        self.name = name
        self.mask_generator = mask_generator
        self.face_detector = face_detector
