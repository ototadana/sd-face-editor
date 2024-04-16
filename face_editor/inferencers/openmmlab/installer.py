import os

import launch
from face_editor.use_cases.installer import Installer


class OpenMMLabInstaller(Installer):
    def name(self) -> str:
        return "OpenMMLab"

    def install(self) -> None:
        launch.run_pip(
            'install openmim "mmsegmentation>=1.0.0" huggingface_hub mmdet',
            "requirements for openmmlab inferencers of Face Editor",
        )
        cmd = "mim"
        if os.name == "nt":
            cmd = os.path.join("venv", "Scripts", cmd)

        launch.run(f"{cmd} install mmengine")
        launch.run(f'{cmd} install "mmcv>=2.0.0"')
