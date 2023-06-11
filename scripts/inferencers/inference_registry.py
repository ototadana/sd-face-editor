from typing import Dict

from scripts.inferencers.bisenet_mask_generator import BiSeNetMaskGenerator
from scripts.inferencers.ellipse_mask_generator import EllipseMaskGenerator
from scripts.inferencers.lbpcascade_animeface_detector import LbpcascadeAnimefaceDetector
from scripts.inferencers.retinaface_detector import RetinafaceDetector
from scripts.use_cases.inferencer_set import InferencerSet


class InferencerRegistry:
    registry: Dict[str, InferencerSet] = {
        InferencerSet.Name.STANDARD.value: InferencerSet(
            InferencerSet.Name.STANDARD.value, RetinafaceDetector(), BiSeNetMaskGenerator()
        ),
        InferencerSet.Name.ANIME.value: InferencerSet(
            InferencerSet.Name.ANIME.value, LbpcascadeAnimefaceDetector(), EllipseMaskGenerator()
        ),
    }

    @classmethod
    def get(cls, name: str) -> InferencerSet:
        return cls.registry[name]
