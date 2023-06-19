import os

import modules.scripts as scripts


def get_path(name: str) -> str:
    dir = os.path.join(scripts.basedir(), name)
    if not os.path.isdir(dir):
        dir = os.path.join(scripts.basedir(), "extensions", "sd-face-editor", name)
        if not os.path.isdir(dir):
            raise RuntimeError(f"not found:{dir}")
    return dir


workflows_dir = os.path.join(get_path("workflows"))
assets_dir = os.path.join(get_path("assets"))
scripts_dir = os.path.join(get_path("scripts"))
