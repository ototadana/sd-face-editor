try:
    from scripts.io.util import load_classes_from_directory
except Exception:
    import os
    import sys

    sys.path.append(os.path.dirname(__file__))
    from scripts.io.util import load_classes_from_directory

import traceback

from modules import shared

from scripts.use_cases.installer import Installer

for component in shared.opts.data.get("face_editor_additional_components", []):
    for cls in load_classes_from_directory(Installer, True):
        try:
            cls().install()
        except Exception as e:
            print(traceback.format_exc())
            print(f"Face Editor: {e}")
