import os
from typing import Dict

from face_editor.io.util import workflows_dir


class Option:
    DEFAULT_FACE_MARGIN = 1.6
    DEFAULT_CONFIDENCE = 0.97
    DEFAULT_STRENGTH1 = 0.4
    DEFAULT_STRENGTH2 = 0.0
    DEFAULT_MAX_FACE_COUNT = 20
    DEFAULT_MASK_SIZE = 0
    DEFAULT_MASK_BLUR = 12
    DEFAULT_PROMPT_FOR_FACE = ""
    DEFAULT_APPLY_INSIDE_MASK_ONLY = True
    DEFAULT_SAVE_ORIGINAL_IMAGE = False
    DEFAULT_SHOW_ORIGINAL_IMAGE = False
    DEFAULT_SHOW_INTERMEDIATE_STEPS = False
    DEFAULT_APPLY_SCRIPTS_TO_FACES = False
    DEFAULT_FACE_SIZE = 512
    DEFAULT_USE_MINIMAL_AREA = False
    DEFAULT_IGNORE_LARGER_FACES = True
    DEFAULT_AFFECTED_AREAS = ["Face"]
    DEFAULT_WORKFLOW = open(os.path.join(workflows_dir, "default.json")).read()
    DEFAULT_UPSCALER = "None"
    DEFAULT_TILT_ADJUSTMENT_THRESHOLD = 40

    def __init__(self, *args) -> None:
        self.extra_options: Dict[str, Dict[str, str]] = {}
        self.face_margin = Option.DEFAULT_FACE_MARGIN
        self.confidence = Option.DEFAULT_CONFIDENCE
        self.strength1 = Option.DEFAULT_STRENGTH1
        self.strength2 = Option.DEFAULT_STRENGTH2
        self.max_face_count = Option.DEFAULT_MAX_FACE_COUNT
        self.mask_size = Option.DEFAULT_MASK_SIZE
        self.mask_blur = Option.DEFAULT_MASK_BLUR
        self.prompt_for_face = Option.DEFAULT_PROMPT_FOR_FACE
        self.apply_inside_mask_only = Option.DEFAULT_APPLY_INSIDE_MASK_ONLY
        self.save_original_image = Option.DEFAULT_SAVE_ORIGINAL_IMAGE
        self.show_intermediate_steps = Option.DEFAULT_SHOW_INTERMEDIATE_STEPS
        self.apply_scripts_to_faces = Option.DEFAULT_APPLY_SCRIPTS_TO_FACES
        self.face_size = Option.DEFAULT_FACE_SIZE
        self.use_minimal_area = Option.DEFAULT_USE_MINIMAL_AREA
        self.ignore_larger_faces = Option.DEFAULT_IGNORE_LARGER_FACES
        self.affected_areas = Option.DEFAULT_AFFECTED_AREAS
        self.show_original_image = Option.DEFAULT_SHOW_ORIGINAL_IMAGE
        self.workflow = Option.DEFAULT_WORKFLOW
        self.upscaler = Option.DEFAULT_UPSCALER
        self.tilt_adjustment_threshold = Option.DEFAULT_TILT_ADJUSTMENT_THRESHOLD

        if len(args) > 0 and isinstance(args[0], dict):
            self.update_by_dict(args[0])
        else:
            self.update_by_list(args)

        self.apply_scripts_to_faces = False

    def update_by_list(self, args: tuple) -> None:
        arg_len = len(args)
        self.face_margin = args[0] if arg_len > 0 and isinstance(args[0], (float, int)) else self.face_margin
        self.confidence = args[1] if arg_len > 1 and isinstance(args[1], (float, int)) else self.confidence
        self.strength1 = args[2] if arg_len > 2 and isinstance(args[2], (float, int)) else self.strength1
        self.strength2 = args[3] if arg_len > 3 and isinstance(args[3], (float, int)) else self.strength2
        self.max_face_count = args[4] if arg_len > 4 and isinstance(args[4], int) else self.max_face_count
        self.mask_size = args[5] if arg_len > 5 and isinstance(args[5], int) else self.mask_size
        self.mask_blur = args[6] if arg_len > 6 and isinstance(args[6], int) else self.mask_blur
        self.prompt_for_face = args[7] if arg_len > 7 and isinstance(args[7], str) else self.prompt_for_face
        self.apply_inside_mask_only = (
            args[8] if arg_len > 8 and isinstance(args[8], bool) else self.apply_inside_mask_only
        )
        self.save_original_image = args[9] if arg_len > 9 and isinstance(args[9], bool) else self.save_original_image
        self.show_intermediate_steps = (
            args[10] if arg_len > 10 and isinstance(args[10], bool) else self.show_intermediate_steps
        )
        self.apply_scripts_to_faces = (
            args[11] if arg_len > 11 and isinstance(args[11], bool) else self.apply_scripts_to_faces
        )
        self.face_size = args[12] if arg_len > 12 and isinstance(args[12], int) else self.face_size
        self.use_minimal_area = args[13] if arg_len > 13 and isinstance(args[13], bool) else self.use_minimal_area
        self.ignore_larger_faces = args[14] if arg_len > 14 and isinstance(args[14], bool) else self.ignore_larger_faces
        self.affected_areas = args[15] if arg_len > 15 and isinstance(args[15], list) else self.affected_areas
        self.show_original_image = args[16] if arg_len > 16 and isinstance(args[16], bool) else self.show_original_image
        self.workflow = args[17] if arg_len > 17 and isinstance(args[17], str) else self.workflow
        self.upscaler = args[18] if arg_len > 18 and isinstance(args[18], str) else self.upscaler
        self.tilt_adjustment_threshold = (
            args[19] if arg_len > 19 and isinstance(args[19], int) else self.tilt_adjustment_threshold
        )

    def update_by_dict(self, params: dict) -> None:
        self.face_margin = params.get("face_margin", self.face_margin)
        self.confidence = params.get("confidence", self.confidence)
        self.strength1 = params.get("strength1", self.strength1)
        self.strength2 = params.get("strength2", self.strength2)
        self.max_face_count = params.get("max_face_count", self.max_face_count)
        self.mask_size = params.get("mask_size", self.mask_size)
        self.mask_blur = params.get("mask_blur", self.mask_blur)
        self.prompt_for_face = params.get("prompt_for_face", self.prompt_for_face)
        self.apply_inside_mask_only = params.get("apply_inside_mask_only", self.apply_inside_mask_only)
        self.save_original_image = params.get("save_original_image", self.save_original_image)
        self.show_intermediate_steps = params.get("show_intermediate_steps", self.show_intermediate_steps)
        self.apply_scripts_to_faces = params.get("apply_scripts_to_faces", self.apply_scripts_to_faces)
        self.face_size = params.get("face_size", self.face_size)
        self.use_minimal_area = params.get("use_minimal_area", self.use_minimal_area)
        self.ignore_larger_faces = params.get("ignore_larger_faces", self.ignore_larger_faces)
        self.affected_areas = params.get("affected_areas", self.affected_areas)
        self.show_original_image = params.get("show_original_image", self.show_original_image)
        self.workflow = params.get("workflow", self.workflow)
        self.upscaler = params.get("upscaler", self.upscaler)
        self.tilt_adjustment_threshold = params.get("tilt_adjustment_threshold", self.tilt_adjustment_threshold)

        for k, v in params.items():
            if isinstance(v, dict):
                self.extra_options[k] = v

    def to_dict(self) -> dict:
        d = {
            Option.add_prefix("enabled"): True,
            Option.add_prefix("face_margin"): self.face_margin,
            Option.add_prefix("confidence"): self.confidence,
            Option.add_prefix("strength1"): self.strength1,
            Option.add_prefix("strength2"): self.strength2,
            Option.add_prefix("max_face_count"): self.max_face_count,
            Option.add_prefix("mask_size"): self.mask_size,
            Option.add_prefix("mask_blur"): self.mask_blur,
            Option.add_prefix("prompt_for_face"): self.prompt_for_face if len(self.prompt_for_face) > 0 else '""',
            Option.add_prefix("apply_inside_mask_only"): self.apply_inside_mask_only,
            Option.add_prefix("apply_scripts_to_faces"): self.apply_scripts_to_faces,
            Option.add_prefix("face_size"): self.face_size,
            Option.add_prefix("use_minimal_area"): self.use_minimal_area,
            Option.add_prefix("ignore_larger_faces"): self.ignore_larger_faces,
            Option.add_prefix("affected_areas"): str.join(";", self.affected_areas),
            Option.add_prefix("workflow"): self.workflow,
            Option.add_prefix("upscaler"): self.upscaler,
            Option.add_prefix("tilt_adjustment_threshold"): self.tilt_adjustment_threshold,
        }

        for option_group_name, options in self.extra_options.items():
            prefix = Option.add_prefix(option_group_name)
            for k, v in options.items():
                d[f"{prefix}_{k}"] = v

        return d

    def add_options(self, option_group_name: str, options: Dict[str, str]):
        self.extra_options[option_group_name] = options

    @staticmethod
    def add_prefix(text: str) -> str:
        return "face_editor_" + text
