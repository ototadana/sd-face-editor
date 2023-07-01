from modules import shared

from scripts.io.util import load_classes_from_directory
from scripts.use_cases.installer import Installer

for component in shared.opts.data.get("face_editor_additional_components", []):
    for cls in load_classes_from_directory(Installer, True):
        try:
            cls().install()
        except Exception as e:
            print(f"Face Editor: {e}")
