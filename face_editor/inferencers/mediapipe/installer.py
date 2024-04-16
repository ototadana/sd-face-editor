from typing import List

from face_editor.use_cases.installer import Installer


class MediaPipeInstaller(Installer):
    def name(self) -> str:
        return "MediaPipe"

    def requirements(self) -> List[str]:
        return ["mediapipe"]
