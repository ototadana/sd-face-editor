from typing import List

from scripts.use_cases.installer import Installer


class InsightFaceInstaller(Installer):
    def name(self) -> str:
        return "InsightFace"

    def requirements(self) -> List[str]:
        return ['"insightface>=0.7.3"', "onnxruntime"]

    def install(self) -> None:
        try:
            from scripts.inferencers.insightface.detector import InsightFaceDetector

            InsightFaceDetector()
        except Exception:
            super().install()
        return None
