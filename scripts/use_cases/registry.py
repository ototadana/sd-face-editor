import traceback
from typing import Dict

from scripts.io.util import load_classes_from_directory
from scripts.use_cases.face_detector import FaceDetector
from scripts.use_cases.face_processor import FaceProcessor
from scripts.use_cases.frame_editor import FrameEditor
from scripts.use_cases.mask_generator import MaskGenerator


def create(all_classes, type: str) -> Dict:
    d = {}
    for cls in all_classes:
        try:
            c = cls()
            d[c.name().lower()] = c
        except Exception as e:
            print(traceback.format_exc())
            print(f"Face Editor: {cls}, Error: {e}")
    return d


def load_face_detector() -> Dict[str, FaceDetector]:
    return create(load_classes_from_directory(FaceDetector), "FaceDetector")


def load_face_processor() -> Dict[str, FaceProcessor]:
    return create(load_classes_from_directory(FaceProcessor), "FaceProcessor")


def load_mask_generator() -> Dict[str, MaskGenerator]:
    return create(load_classes_from_directory(MaskGenerator), "MaskGenerator")


def load_frame_editor() -> Dict[str, FrameEditor]:
    return create(load_classes_from_directory(FrameEditor), "FrameEditor")


face_detectors = load_face_detector()
face_processors = load_face_processor()
mask_generators = load_mask_generator()
frame_editors = load_frame_editor()

face_detector_names = list(face_detectors.keys())
face_processor_names = list(face_processors.keys())
mask_generator_names = list(mask_generators.keys())
frame_editor_names = list(frame_editors.keys())
