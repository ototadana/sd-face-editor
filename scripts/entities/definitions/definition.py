from abc import ABC, abstractmethod
from typing import Any, Dict, TypeVar

T = TypeVar("T", bound="Definition")


class Definition(ABC):
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def from_dict(self, value: Dict[str, Any]) -> T:  # type: ignore
        pass
