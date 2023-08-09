import launch

from scripts.use_cases.installer import Installer


class OpenMMLabInstaller(Installer):
    def name(self) -> str:
        return "OpenMMLab"

    def install(self) -> None:
        launch.run_pip(
            'install openmim "mmsegmentation>=1.0.0" huggingface_hub mmdet',
            "requirements for openmmlab inferencers of Face Editor",
        )
        launch.run("mim install mmengine")
        launch.run('mim install "mmcv>=2.0.0"')
