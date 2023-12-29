from typing import List

import cv2
import modules.shared as shared
import numpy as np
from modules.processing import StableDiffusionProcessingImg2Img
from PIL.Image import Image

from scripts.entities.definitions import Job, Rule, Worker, Workflow
from scripts.entities.face import Face
from scripts.entities.option import Option
from scripts.entities.rect import Rect
from scripts.entities.settings import Settings
from scripts.use_cases import condition_matcher, query_matcher, registry
from scripts.use_cases.image_processing_util import rotate_array, rotate_image


class WorkflowManager:
    @classmethod
    def get(cls, workflow: str) -> "WorkflowManager":
        manager = cls(Workflow.parse_raw(workflow))

        for face_detector in manager.workflow.face_detector:
            if face_detector.name not in registry.face_detector_names:
                raise KeyError(f"face_detector `{face_detector.name}` does not exist")

        rules = manager.workflow.rules if manager.workflow.rules is not None else []
        for rule in rules:
            for job in rule.then:
                if job.face_processor.name not in registry.face_processor_names:
                    raise KeyError(f"face_processor `{job.face_processor.name}` does not exist")
                if job.mask_generator.name not in registry.mask_generator_names:
                    raise KeyError(f"mask_generator `{job.mask_generator.name}` does not exist")
            if rule.when is not None and rule.when.tag is not None and "?" in rule.when.tag:
                _, query = condition_matcher.parse_tag(rule.when.tag)
                if len(query) > 0:
                    query_matcher.validate(query)

        if manager.workflow.postprocessors is None or len(manager.workflow.postprocessors) == 0:
            manager.workflow.postprocessors = [Worker(name="Img2Img", params={"strength": 0})]

        manager.__validate_frame_editors(manager.workflow.preprocessors)
        manager.__validate_frame_editors(manager.workflow.postprocessors)

        return manager

    def __validate_frame_editors(cls, frame_editors: List[Worker]) -> List[Worker]:
        if frame_editors is None:
            return

        for frame_editor in frame_editors:
            if frame_editor.name not in registry.frame_editor_names:
                raise KeyError(f"frame_editor `{frame_editor.name}` does not exist")

    def __init__(self, workflow: Workflow) -> None:
        self.workflow = workflow
        self.correct_tilt = Settings.correct_tilt()

    def detect_faces(self, image: Image, option: Option) -> List[Rect]:
        results = []

        for fd in self.workflow.face_detector:
            face_detector = registry.face_detectors[fd.name]
            params = fd.params.copy()
            params["option"] = option
            params["confidence"] = option.confidence
            results.extend(face_detector.detect_faces(image, **params))

        return results

    def select_rule(self, faces: List[Face], index: int, width: int, height: int) -> Rule:
        face = faces[index]
        if face.face_area is None:
            return None

        rules = self.workflow.rules if self.workflow.rules is not None else []
        for rule in rules:
            if rule.when is None:
                return rule
            if condition_matcher.check_condition(rule.when, faces, face, width, height):
                return rule

        return None

    def process(self, jobs: List[Job], face: Face, p: StableDiffusionProcessingImg2Img, option: Option) -> Image:
        if len(jobs) == 0:
            return None

        for job in jobs:
            fp = job.face_processor
            face_processor = registry.face_processors[fp.name]
            params = fp.params.copy()
            params["option"] = option
            params["strength1"] = option.strength1

            angle = face.get_angle()
            correct_tilt = self.__correct_tilt(option, angle)
            face.image = rotate_image(face.image, angle) if correct_tilt else face.image

            image = face_processor.process(face, p, **params)

            face.image = rotate_image(image, -angle) if correct_tilt else image
        return face.image

    def has_preprocessor(self) -> bool:
        return self.workflow.preprocessors is not None and len(self.workflow.preprocessors) > 0

    def has_postprocessor(self) -> bool:
        return self.workflow.postprocessors is not None and len(self.workflow.postprocessors) > 0

    def preprocess(
        self, p: StableDiffusionProcessingImg2Img, faces: List[Face], option: Option, output_images: List[Image]
    ) -> None:
        if self.workflow.preprocessors is None:
            return

        self.__edit(self.workflow.preprocessors, p, faces, option, output_images)
        if p.init_images[0].width == p.width and p.init_images[0].height == p.height:
            return

        default_size = 1024 if getattr(shared.sd_model, "is_sdxl", False) else 512
        if not self.__has_resize_tool(self.workflow.preprocessors) and (
            max(p.init_images[0].width, p.init_images[0].height) < default_size
        ):
            resize_tool = registry.frame_editors["resize"]
            if p.init_images[0].width > p.init_images[0].height:
                p.width = default_size
                p.height = round(default_size * p.init_images[0].height / p.init_images[0].width)
            else:
                p.width = round(default_size * p.init_images[0].width / p.init_images[0].height)
                p.height = default_size
            resize_tool.edit(p, faces, output_images, width=p.width, height=p.height, resize_mode=1)

    def postprocess(
        self, p: StableDiffusionProcessingImg2Img, faces: List[Face], option: Option, output_images: List[Image]
    ) -> None:
        if self.workflow.postprocessors is None:
            return

        if option.show_intermediate_steps:
            output_images.append(p.init_images[0])

        self.__edit(self.workflow.postprocessors, p, faces, option, output_images)

    def __edit(
        self,
        frame_editors: List[Worker],
        p: StableDiffusionProcessingImg2Img,
        faces: List[Face],
        option: Option,
        output_images: List[Image],
    ):
        output_images = output_images if option.show_intermediate_steps else None
        for frame_editor in frame_editors:
            print(f"frame_editor: {frame_editor.name}")
            fe = registry.frame_editors[frame_editor.name]
            params = frame_editor.params.copy()
            params["option"] = option
            fe.edit(p, faces, output_images, **params)

    def __has_resize_tool(self, frame_editors: List[Worker]) -> bool:
        for frame_editor in frame_editors:
            if frame_editor.name == "resize":
                return True
        return False

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
            params["face_area_total_pixels"] = face.face_area_total_pixels

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
