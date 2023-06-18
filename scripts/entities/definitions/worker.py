from typing import Any, Dict

from scripts.entities.definitions.definition import Definition


class Worker(Definition):
    def __init__(self, name: str = "", params: Dict[str, Any] = {}) -> None:
        self.name = name
        self.params = params

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "params": self.params}

    def from_dict(self, value: Dict[str, Any]) -> "Worker":
        self.name = value.get("name", "")
        self.params = value.get("params", {})
        return self
