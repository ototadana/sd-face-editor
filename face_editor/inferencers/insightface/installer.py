from typing import List

from face_editor.use_cases.installer import Installer


class InsightFaceInstaller(Installer):
    def name(self) -> str:
        return "InsightFace"

    def requirements(self) -> List[str]:
        return ['"insightface>=0.7.3"', "onnxruntime"]

    def install(self) -> None:
        try:
            from face_editor.inferencers.insightface.detector import InsightFaceDetector

            InsightFaceDetector()
        except Exception:
            super().install()
        return None
