from typing import List

from scripts.use_cases.installer import Installer


class InsightFaceInstaller(Installer):
    def name(self) -> str:
        return "InsightFace"

    def requirements(self) -> List[str]:
        return ["insightface"]
