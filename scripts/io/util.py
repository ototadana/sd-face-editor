import importlib.util
import inspect
import os
from typing import List, Type

import modules.scripts as scripts


def get_path(*p: str) -> str:
    dir = os.path.join(scripts.basedir(), *p)
    if not os.path.isdir(dir):
        dir = os.path.join(scripts.basedir(), "extensions", "sd-face-editor", *p)
        if not os.path.isdir(dir):
            raise RuntimeError(f"not found:{dir}")
    return dir


workflows_dir = os.path.join(get_path("workflows"))
assets_dir = os.path.join(get_path("assets"))
inferencers_dir = os.path.join(get_path("scripts", "inferencers"))


def load_classes_from_file(file_path: str, base_class: Type) -> List[Type]:
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    classes = []

    try:
        spec.loader.exec_module(module)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, base_class) and cls is not base_class:
                classes.append(cls)
    except Exception as e:
        print(file_path, ":", e)

    return classes


def load_classes_from_directory(base_class: Type) -> List[Type]:
    all_classes = []
    for file in os.listdir(inferencers_dir):
        if file.endswith(".py") and file != os.path.basename(__file__):
            file_path = os.path.join(inferencers_dir, file)
            classes = load_classes_from_file(file_path, base_class)
            if classes:
                print(f"Successfully loaded {len(classes)} classes from {file_path}")
                all_classes.extend(classes)
    return all_classes
