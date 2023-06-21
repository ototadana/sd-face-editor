from scripts.io.util import load_classes_from_directory
from scripts.use_cases.installer import Installer

for cls in load_classes_from_directory(Installer):
    cls().install()
