from typing import List, Tuple

import cv2
import numpy as np
from face_editor.entities.definitions import Condition, Job, Rule, Workflow
from face_editor.entities.face import Face
from face_editor.entities.option import Option
from face_editor.entities.rect import Rect
from face_editor.use_cases import query_matcher, registry
from face_editor.use_cases.image_processing_util import rotate_array, rotate_image
from modules import shared
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image


class WorkflowManager:
    @classmethod
    def get(cls, workflow: str) -> "WorkflowManager":
        manager = cls(Workflow.parse_raw(workflow))

        for face_detector in manager.workflow.face_detector:
            if face_detector.name not in registry.face_detector_names:
                raise KeyError(f"face_detector `{face_detector.name}` does not exist")

        for rule in manager.workflow.rules:
            for job in rule.then:
                if job.face_processor.name not in registry.face_processor_names:
                    raise KeyError(f"face_processor `{job.face_processor.name}` does not exist")
                if job.mask_generator.name not in registry.mask_generator_names:
                    raise KeyError(f"mask_generator `{job.mask_generator.name}` does not exist")
            if rule.when is not None and rule.when.tag is not None and "?" in rule.when.tag:
                _, query = cls.__parse_tag(rule.when.tag)
                if len(query) > 0:
                    query_matcher.validate(query)

        return manager

    def __init__(self, workflow: Workflow) -> None:
        self.workflow = workflow
        self.correct_tilt = shared.opts.data.get("face_editor_correct_tilt", False)

    def detect_faces(self, image: Image, option: Option) -> List[Rect]:
        results = []

        for fd in self.workflow.face_detector:
            face_detector = registry.face_detectors[fd.name]
            params = fd.params.copy()
            params["confidence"] = option.confidence
            results.extend(face_detector.detect_faces(image, **params))

        return results

    def select_rule(self, faces: List[Face], index: int, width: int, height: int) -> Rule:
        face = faces[index]
        if face.face_area is None:
            return None

        for rule in self.workflow.rules:
            if rule.when is None:
                return rule

            if self.__is_tag_match(rule.when, face):
                tag_matched_faces = [f for f in faces if self.__is_tag_match(rule.when, f)]
                if self.__is_criteria_match(rule.when, tag_matched_faces, face, width, height):
                    return rule

        return None

    @classmethod
    def __parse_tag(cls, tag: str) -> Tuple[str, str]:
        parts = tag.split("?", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    def __is_tag_match(self, condition: Condition, face: Face) -> bool:
        if condition.tag is None or len(condition.tag) == 0:
            return True

        condition_tag = condition.tag.lower()
        if condition_tag == "any":
            return True

        tag, query = self.__parse_tag(condition_tag)
        face_tag = face.face_area.tag.lower() if face.face_area.tag is not None else ""
        if tag != face_tag:
            return False
        if len(query) == 0:
            return True
        return query_matcher.evaluate(query, face.face_area.attributes)

    def __is_criteria_match(self, condition: Condition, faces: List[Face], face: Face, width: int, height: int) -> bool:
        if not condition.has_criteria():
            return True

        criteria = condition.get_criteria()
        indices = condition.get_indices()

        if criteria == "all":
            return True

        if criteria in {"left", "leftmost"}:
            return self.__is_left(indices, faces, face)
        if criteria in {"center", "center_horizontal", "middle_horizontal"}:
            return self.__is_center(indices, faces, face, width)
        if criteria in {"right", "rightmost"}:
            return self.__is_right(indices, faces, face)
        if criteria in {"top", "upper", "upmost"}:
            return self.__is_top(indices, faces, face)
        if criteria in {"middle", "center_vertical", "middle_vertical"}:
            return self.__is_middle(indices, faces, face, height)
        if criteria in {"bottom", "lower", "downmost"}:
            return self.__is_bottom(indices, faces, face)
        if criteria in {"small", "tiny", "smaller"}:
            return self.__is_small(indices, faces, face)
        if criteria in {"large", "big", "bigger"}:
            return self.__is_large(indices, faces, face)
        return False

    def __is_left(self, indices: List[int], faces: List[Face], face: Face) -> bool:
        sorted_faces = sorted(faces, key=lambda f: f.face_area.left)
        return sorted_faces.index(face) in indices

    def __is_center(self, indices: List[int], faces: List[Face], face: Face, width: int) -> bool:
        sorted_faces = sorted(faces, key=lambda f: abs((f.face_area.center - width / 2)))
        return sorted_faces.index(face) in indices

    def __is_right(self, indices: List[int], faces: List[Face], face: Face) -> bool:
        sorted_faces = sorted(faces, key=lambda f: f.face_area.right, reverse=True)
        return sorted_faces.index(face) in indices

    def __is_top(self, indices: List[int], faces: List[Face], face: Face) -> bool:
        sorted_faces = sorted(faces, key=lambda f: f.face_area.top)
        return sorted_faces.index(face) in indices

    def __is_middle(self, indices: List[int], faces: List[Face], face: Face, height: int) -> bool:
        sorted_faces = sorted(faces, key=lambda f: abs(f.face_area.middle - height / 2))
        return sorted_faces.index(face) in indices

    def __is_bottom(self, indices: List[int], faces: List[Face], face: Face) -> bool:
        sorted_faces = sorted(faces, key=lambda f: f.face_area.bottom, reverse=True)
        return sorted_faces.index(face) in indices

    def __is_small(self, indices: List[int], faces: List[Face], face: Face) -> bool:
        sorted_faces = sorted(faces, key=lambda f: f.face_area.size)
        return sorted_faces.index(face) in indices

    def __is_large(self, indices: List[int], faces: List[Face], face: Face) -> bool:
        sorted_faces = sorted(faces, key=lambda f: f.face_area.size, reverse=True)
        return sorted_faces.index(face) in indices

    def process(self, jobs: List[Job], face: Face, p: StableDiffusionProcessingImg2Img, option: Option) -> Image:
        for job in jobs:
            fp = job.face_processor
            face_processor = registry.face_processors[fp.name]
            params = fp.params.copy()
            params["strength1"] = option.strength1

            angle = face.get_angle()
            correct_tilt = self.__correct_tilt(option, angle)
            face.image = rotate_image(face.image, angle) if correct_tilt else face.image

            image = face_processor.process(face, p, **params)

            face.image = rotate_image(image, -angle) if correct_tilt else image
        return face.image

    def __correct_tilt(self, option: Option, angle: float) -> bool:
        if self.correct_tilt:
            return True
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
        return angle > option.tilt_adjustment_threshold

    def generate_mask(self, jobs: List[Job], face_image: np.ndarray, face: Face, option: Option) -> np.ndarray:
        mask = None
        for job in jobs:
            mg = job.mask_generator
            mask_generator = registry.mask_generators[mg.name]
            params = mg.params.copy()
            params["mask_size"] = option.mask_size
            params["use_minimal_area"] = option.use_minimal_area
            params["affected_areas"] = option.affected_areas
            params["tag"] = face.face_area.tag

            angle = face.get_angle()
            correct_tilt = self.__correct_tilt(option, angle)
            image = rotate_array(face_image, angle) if correct_tilt else face_image
            face_area_on_image = face.rotate_face_area_on_image(angle) if correct_tilt else face.face_area_on_image
            m = mask_generator.generate_mask(image, face_area_on_image, **params)
            m = rotate_array(m, -angle) if correct_tilt else m

            if mask is None:
                mask = m
            else:
                mask = mask + m

        assert mask is not None
        if option.mask_blur > 0:
            mask = cv2.blur(mask, (option.mask_blur, option.mask_blur))

        return mask
