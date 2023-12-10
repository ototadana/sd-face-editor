from typing import List

from modules import shared

from scripts.entities.option import Option


class Settings:
    SEARCH_SUBDIRECTORIES = "face_editor_search_subdirectories"
    ADDITIONAL_COMPONENTS = "face_editor_additional_components"
    SAVE_ORIGINAL_ON_DETECTION_FAIL = "face_editor_save_original_on_detection_fail"
    CORRECT_TILT = "face_editor_correct_tilt"
    AUTO_FACE_SIZE_BY_MODEL = "face_editor_auto_face_size_by_model"
    DEFAULT_UPSCALER = "face_editor_default_upscaler"
    SCRIPT_INDEX = "face_editor_script_index"

    @classmethod
    def search_subdirectories(cls) -> bool:
        return shared.opts.data.get(cls.SEARCH_SUBDIRECTORIES, False)

    @classmethod
    def additional_components(cls) -> List[str]:
        return shared.opts.data.get(cls.ADDITIONAL_COMPONENTS, [])

    @classmethod
    def save_original_on_detection_fail(cls) -> bool:
        return shared.opts.data.get(cls.SAVE_ORIGINAL_ON_DETECTION_FAIL, False)

    @classmethod
    def correct_tilt(cls) -> bool:
        return shared.opts.data.get(cls.CORRECT_TILT, False)

    @classmethod
    def auto_face_size_by_model(cls) -> bool:
        return shared.opts.data.get(cls.AUTO_FACE_SIZE_BY_MODEL, False)

    @classmethod
    def default_upscaler(cls) -> str:
        return shared.opts.data.get(cls.DEFAULT_UPSCALER, Option.DEFAULT_UPSCALER)

    @classmethod
    def script_index(cls) -> int:
        return shared.opts.data.get(cls.SCRIPT_INDEX, 99)
