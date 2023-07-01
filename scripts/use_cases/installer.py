from abc import ABC, abstractmethod
from typing import List

import launch


class Installer(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    def requirements(self) -> List[str]:
        return []

    def install(self) -> None:
        requirements_to_install = [req for req in self.requirements() if not launch.is_installed(req)]
        if len(requirements_to_install) > 0:
            launch.run_pip(
                f"install {' '.join(requirements_to_install)}",
                f"requirements for {self.name()}: {requirements_to_install}",
            )
