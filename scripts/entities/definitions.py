from typing import Dict, List, Optional, Union

from pydantic import BaseModel, root_validator, validator


class Worker(BaseModel):
    name: str
    params: Optional[Dict]

    @root_validator(pre=True)
    def default_params(cls, values):
        if "params" not in values or values["params"] is None:
            values["params"] = {}
        return values

    @validator("name")
    def lowercase_name(cls, v):
        return v.lower()


def parse_worker_field(value: Union[str, Dict, Worker]) -> Worker:
    if isinstance(value, Dict):
        return Worker(**value)
    if isinstance(value, str):
        return Worker(name=value)
    return value


def validate_frame_editors(value) -> List[Worker]:
    if value is None:
        return []
    elif isinstance(value, List):
        return [parse_worker_field(item) for item in value]
    else:
        return [parse_worker_field(value)]


class Condition(BaseModel):
    tag: Optional[str]
    criteria: Optional[str]

    def get_indices(self) -> List[int]:
        if self.criteria is None or ":" not in self.criteria:
            return [0]

        indices: List[int] = []
        for part in self.criteria.split(":")[1].split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, [x.strip() for x in part.split("-")])
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))

        return indices

    def get_criteria(self) -> str:
        if self.criteria is None or self.criteria == "":
            return ""
        return self.criteria.split(":")[0].strip().lower()

    def has_criteria(self) -> bool:
        return len(self.get_criteria()) > 0

    def is_all(self) -> bool:
        return self.get_criteria() == "all"

    def is_left(self) -> bool:
        return self.get_criteria() in {"left", "leftmost"}

    def is_center(self) -> bool:
        return self.get_criteria() in {
            "center",
            "center_horizontal",
            "middle_horizontal",
        }

    def is_right(self) -> bool:
        return self.get_criteria() in {"right", "rightmost"}

    def is_top(self) -> bool:
        return self.get_criteria() in {"top", "upper", "upmost"}

    def is_middle(self) -> bool:
        return self.get_criteria() in {"middle", "center_vertical", "middle_vertical"}

    def is_bottom(self) -> bool:
        return self.get_criteria() in {"bottom", "lower", "downmost", "bottommost"}

    def is_small(self) -> bool:
        return self.get_criteria() in {"small", "tiny", "smaller"}

    def is_large(self) -> bool:
        return self.get_criteria() in {"large", "big", "bigger"}

    @validator("criteria")
    def validate_criteria(cls, value):
        c = cls()
        c.criteria = value
        c.get_indices()
        return value


class Job(BaseModel):
    face_processor: Worker
    mask_generator: Worker

    @validator("face_processor", "mask_generator", pre=True)
    def parse_worker_fields(cls, value):
        return parse_worker_field(value)


class Rule(BaseModel):
    when: Optional[Condition] = None
    then: Union[Job, List[Job]]

    @validator("then", pre=True)
    def parse_jobs(cls, value):
        if isinstance(value, Dict):
            return [Job.parse_obj(value)]
        elif isinstance(value, List):
            return [Job.parse_obj(job) for job in value]


class Workflow(BaseModel):
    face_detector: Union[Worker, List[Worker]]
    rules: Optional[Union[Rule, List[Rule]]]
    preprocessors: Optional[Union[Worker, List[Worker]]]
    postprocessors: Optional[Union[Worker, List[Worker]]]

    @validator("face_detector", pre=True)
    def parse_face_detector(cls, value):
        if isinstance(value, List):
            return [parse_worker_field(item) for item in value]
        else:
            return [parse_worker_field(value)]

    @validator("rules", pre=True)
    def wrap_rule_in_list(cls, value):
        if value is None:
            return []
        elif not isinstance(value, List):
            return [value]
        return value

    @validator("preprocessors", pre=True)
    def parse_preprocessors(cls, value):
        return validate_frame_editors(value)

    @validator("postprocessors", pre=True)
    def parse_postprocessors(cls, value):
        return validate_frame_editors(value)
