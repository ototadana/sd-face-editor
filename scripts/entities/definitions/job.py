from typing import Any, Dict

from scripts.entities.definitions.definition import Definition
from scripts.entities.definitions.worker import Worker


class Job(Definition):
    def __init__(self, name: str = "", face_processor: Worker = None, mask_generator: Worker = None) -> None:
        self.name = name
        self.face_processor = face_processor
        self.mask_generator = mask_generator

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "face_processor": self.face_processor.to_dict() if self.face_processor is not None else {},
            "mask_generator": self.mask_generator.to_dict() if self.mask_generator is not None else {},
        }

    def from_dict(self, value: Dict[str, Any]) -> "Job":
        self.name = value.get("name", "")
        face_processor_data = value.get("face_processor", None)
        if face_processor_data is not None:
            self.face_processor = Worker().from_dict(face_processor_data)
        mask_generator_data = value.get("mask_generator", None)
        if mask_generator_data is not None:
            self.mask_generator = Worker().from_dict(mask_generator_data)
        return self
