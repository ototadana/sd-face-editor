from typing import Dict

from scripts.io.util import load_classes_from_directory
from scripts.use_cases.face_detector import FaceDetector
from scripts.use_cases.face_processor import FaceProcessor
from scripts.use_cases.mask_generator import MaskGenerator


def load_face_detector() -> Dict[str, FaceDetector]:
    all_classes = load_classes_from_directory(FaceDetector)
    return {c.name(): c for cls in all_classes for c in [cls()]}


def load_face_processor() -> Dict[str, FaceProcessor]:
    all_classes = load_classes_from_directory(FaceProcessor)
    return {c.name(): c for cls in all_classes for c in [cls()]}


def load_mask_generator() -> Dict[str, MaskGenerator]:
    all_classes = load_classes_from_directory(MaskGenerator)
    return {c.name(): c for cls in all_classes for c in [cls()]}


face_detectors = load_face_detector()
face_processors = load_face_processor()
mask_generators = load_mask_generator()
face_detector_names = list(face_detectors.keys())
face_processor_names = list(face_processors.keys())
mask_generator_names = list(mask_generators.keys())
