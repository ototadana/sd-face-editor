import json
from typing import List, Tuple

import cv2
import numpy as np
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image

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

    def select_jobs(self, faces: List[Face], index: int) -> List[Job]:
        jobs = []
        for condition in self.workflow.conditions:
            if condition.tag != "Any" and condition.tag.lower() != faces[index].face_area.tag.lower():
                continue
            # TODO: select by condition.criteria
            jobs.extend(condition.jobs)
        return jobs

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
