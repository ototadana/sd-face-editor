from typing import List

from scripts.use_cases.installer import Installer


class YoloDetectorInstaller(Installer):
    def name(self) -> str:
        return "YOLO"

    def requirements(self) -> List[str]:
        return ["huggingface_hub", "ultralytics"]
