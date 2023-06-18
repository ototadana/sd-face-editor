import importlib.util
import inspect
import json
import os
from typing import Dict, List, Type

from scripts.entities.definitions.workflow import Workflow
from scripts.use_cases.face_detector import FaceDetector
from scripts.use_cases.inferencer_set import InferencerSet
from scripts.use_cases.mask_generator import MaskGenerator

current_directory = os.path.dirname(os.path.realpath(__file__))


def load_classes_from_file(file_path: str, base_class: Type) -> List[Type]:
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    classes = []
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if issubclass(cls, base_class) and cls is not base_class:
            classes.append(cls)
    return classes


def load_classes_from_directory(directory_path: str, base_class: Type) -> List[Type]:
    all_classes = []
    for file in os.listdir(directory_path):
        if file.endswith(".py") and file != os.path.basename(__file__):
            file_path = os.path.join(directory_path, file)
            classes = load_classes_from_file(file_path, base_class)
            if classes:
                print(f"Successfully loaded {len(classes)} classes from {file_path}")
                all_classes.extend(classes)
    return all_classes


def load_face_detector() -> Dict[str, FaceDetector]:
    all_classes = load_classes_from_directory(current_directory, FaceDetector)
    return {c.name(): c for cls in all_classes for c in [cls()]}


def load_mask_generator() -> Dict[str, MaskGenerator]:
    all_classes = load_classes_from_directory(current_directory, MaskGenerator)
    return {c.name(): c for cls in all_classes for c in [cls()]}


face_detectors = load_face_detector()
mask_generators = load_mask_generator()
face_detector_names = list(face_detectors.keys())
mask_generator_names = list(mask_generators.keys())


def get(workflow: str) -> InferencerSet:
    wf = Workflow().from_dict(json.loads(workflow))
    fd = wf.face_detector
    mg = wf.conditions[0].jobs[0].mask_generator
    return InferencerSet(face_detectors[fd.name], fd.params, mask_generators[mg.name], mg.params)
