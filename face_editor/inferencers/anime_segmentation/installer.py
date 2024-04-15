from typing import List

from face_editor.use_cases.installer import Installer


class AnimeSegmentationInstaller(Installer):
    def name(self) -> str:
        return "AnimeSegmentation"

    def requirements(self) -> List[str]:
        return ["huggingface_hub", "onnxruntime"]
