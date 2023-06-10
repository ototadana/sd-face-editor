from scripts.inferencers.bisenet_mask_generator import BiSeNetMaskGenerator
from scripts.inferencers.retinaface_detector import RetinafaceDetector
from scripts.use_cases.inferencer_registry import InferencerRegistry


class InferencerFactory:
    registry: InferencerRegistry = None

    @classmethod
    def create(cls) -> InferencerRegistry:
        if cls.registry is None:
            cls.registry = InferencerRegistry(RetinafaceDetector(), BiSeNetMaskGenerator())
        return cls.registry
