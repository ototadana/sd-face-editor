from typing import Any, Dict, List

from scripts.entities.definitions.condition import Condition
from scripts.entities.definitions.definition import Definition
from scripts.entities.definitions.worker import Worker


class Workflow(Definition):
    def __init__(self, name: str = "", face_detector: Worker = None, conditions: List[Condition] = None) -> None:
        self.name = name
        self.face_detector = face_detector
        self.conditions = conditions if conditions is not None else [Condition()]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "face_detector": self.face_detector.to_dict() if self.face_detector is not None else {},
            "conditions": [condition.to_dict() for condition in self.conditions],
        }

    def from_dict(self, value: Dict[str, Any]) -> "Workflow":
        self.name = value.get("name", "")
        face_detector_data = value.get("face_detector", None)
        if face_detector_data is not None:
            self.face_detector = Worker().from_dict(face_detector_data)
        conditions_data = value.get("conditions", [Condition()])
        self.conditions = [Condition().from_dict(condition_data) for condition_data in conditions_data]
        return self
