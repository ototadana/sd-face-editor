import json
from typing import List, Tuple

import cv2
import numpy as np
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image

from scripts.entities.definitions.condition import Condition
from scripts.entities.definitions.job import Job
from scripts.entities.definitions.workflow import Workflow
from scripts.entities.face import Face
from scripts.entities.option import Option
from scripts.entities.rect import Rect
from scripts.use_cases import registry


class WorkflowManager:
    @classmethod
    def get(cls, workflow: str) -> "WorkflowManager":
        return cls(Workflow().from_dict(json.loads(workflow)))

    def __init__(self, workflow: Workflow) -> None:
        self.workflow = workflow

        fd = workflow.face_detector
        self.face_detector = registry.face_detectors[fd.name]
        self.face_detector_params = fd.params

    def detect_faces(self, image: Image, option: Option) -> List[Rect]:
        fd = self.workflow.face_detector

        face_detector = registry.face_detectors[fd.name]
        params = fd.params.copy()
        params["confidence"] = option.confidence

        return face_detector.detect_faces(image, **params)

    def select_jobs(self, faces: List[Face], index: int, width: int, height: int) -> List[Job]:
        jobs = []
        for condition in self.workflow.conditions:
            if faces[index].face_area is None:
                continue
            if not self.__is_tag_match(condition, faces[index]):
                continue
            if not self.__is_criteria_match(condition, faces, index, width, height):
                continue
            jobs.extend(condition.jobs)
        return jobs

    def __is_tag_match(self, condition: Condition, face: Face) -> bool:
        if condition.tag is None or len(condition.tag) == 0:
            return True

        condition_tag = condition.tag.lower()
        if condition_tag == "any":
            return True

        face_tag = face.face_area.tag.lower() if face.face_area.tag is not None else ""
        return condition_tag == face_tag

    def __is_criteria_match(self, condition: Condition, faces: List[Face], index: int, width: int, height: int) -> bool:
        num = condition.num
        if num < 1:
            return False

        if condition.criteria is None or len(condition.criteria) == 0:
            return True

        criteria = condition.criteria.lower()
        if criteria in {"any", "all"}:
            return True

        if criteria in {"left", "leftmost"}:
            return self.__is_left(num, faces, index)
        if criteria in {"center", "center_horizontal", "middle_horizontal"}:
            return self.__is_center(num, faces, index, width)
        if criteria in {"right", "rightmost"}:
            return self.__is_right(num, faces, index)
        if criteria in {"top", "upper", "upmost"}:
            return self.__is_top(num, faces, index)
        if criteria in {"middle", "center_vertical", "middle_vertical"}:
            return self.__is_middle(num, faces, index, height)
        if criteria in {"bottom", "lower", "downmost"}:
            return self.__is_bottom(num, faces, index)
        if criteria in {"small", "tiny", "smaller"}:
            return self.__is_small(num, faces, index)
        if criteria in {"large", "big", "bigger"}:
            return self.__is_large(num, faces, index)
        return False

    def __is_left(self, num: int, faces: List[Face], index: int) -> bool:
        sorted_faces = sorted(faces, key=lambda f: f.face_area.left)
        return sorted_faces.index(faces[index]) < num

    def __is_center(self, num: int, faces: List[Face], index: int, width: int) -> bool:
        sorted_faces = sorted(faces, key=lambda f: abs((f.face_area.center - width / 2)))
        return sorted_faces.index(faces[index]) < num

    def __is_right(self, num: int, faces: List[Face], index: int) -> bool:
        sorted_faces = sorted(faces, key=lambda f: f.face_area.right, reverse=True)
        return sorted_faces.index(faces[index]) < num

    def __is_top(self, num: int, faces: List[Face], index: int) -> bool:
        sorted_faces = sorted(faces, key=lambda f: f.face_area.top)
        return sorted_faces.index(faces[index]) < num

    def __is_middle(self, num: int, faces: List[Face], index: int, height: int) -> bool:
        sorted_faces = sorted(faces, key=lambda f: abs(f.face_area.middle - height / 2))
        return sorted_faces.index(faces[index]) < num

    def __is_bottom(self, num: int, faces: List[Face], index: int) -> bool:
        sorted_faces = sorted(faces, key=lambda f: f.face_area.bottom, reverse=True)
        return sorted_faces.index(faces[index]) < num

    def __is_small(self, num: int, faces: List[Face], index: int) -> bool:
        sorted_faces = sorted(faces, key=lambda f: f.face_area.size)
        return sorted_faces.index(faces[index]) < num

    def __is_large(self, num: int, faces: List[Face], index: int) -> bool:
        sorted_faces = sorted(faces, key=lambda f: f.face_area.size, reverse=True)
        return sorted_faces.index(faces[index]) < num

    def process(self, jobs: List[Job], face: Face, p: StableDiffusionProcessingImg2Img, option: Option) -> Image:
        for job in jobs:
            fp = job.face_processor
            face_processor = registry.face_processors[fp.name]
            params = fp.params.copy()
            params["strength1"] = option.strength1
            face.image = face_processor.process(face, p, **params)
        return face.image

    def generate_mask(
        self, jobs: List[Job], face_image: np.ndarray, face_area_on_image: Tuple[int, int, int, int], option: Option
    ) -> np.ndarray:
        mask = None
        for job in jobs:
            mg = job.mask_generator
            mask_generator = registry.mask_generators[mg.name]
            params = mg.params.copy()
            params["mask_size"] = option.mask_size
            params["use_minimal_area"] = option.use_minimal_area
            params["affected_areas"] = option.affected_areas
            m = mask_generator.generate_mask(face_image, face_area_on_image, **params)
            if mask is None:
                mask = m
            else:
                mask = mask + m

        assert mask is not None
        if option.mask_blur > 0:
            mask = cv2.blur(mask, (option.mask_blur, option.mask_blur))

        return mask
