from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, root_validator, validator


class Worker(BaseModel):
    name: str
    params: Optional[dict]

    @root_validator(pre=True)
    def default_params(cls, values):
        if "params" not in values or values["params"] is None:
            values["params"] = {}
        return values


def parse_worker_field(value: Union[str, dict, Worker, List[Union[str, dict, Worker]]]) -> Union[Worker, List[Worker]]:
    if isinstance(value, list):
        return [parse_worker_field(item) for item in value]  # type: ignore
    if isinstance(value, dict):
        return Worker(**value)
    if isinstance(value, str):
        return Worker(name=value)
    return value


class Condition(BaseModel):
    tag: Optional[str]
    criteria: Optional[str]
    num: Optional[int]


class Job(BaseModel):
    face_processor: Worker
    mask_generator: Worker

    @root_validator(pre=True)
    def parse_worker_fields(cls, values: Dict[str, Any]):
        for key, value in values.items():
            values[key] = parse_worker_field(value)
        return values


class Rule(BaseModel):
    when: Optional[Condition] = None
    then: Union[Job, List[Job]]

    @root_validator(pre=True)
    def parse_jobs(cls, values: Dict[str, Any]):
        if "then" in values:
            if isinstance(values["then"], dict):
                values["then"] = [Job.parse_obj(values["then"])]
            elif isinstance(values["then"], list):
                values["then"] = [Job.parse_obj(job) for job in values["then"]]
        return values


class Workflow(BaseModel):
    face_detector: Union[Worker, List[Worker]]
    rules: Union[Rule, List[Rule]]

    @validator("face_detector", pre=True)
    def parse_face_detector(cls, value):
        return [parse_worker_field(value)]

    @validator("rules", pre=True)
    def wrap_rule_in_list(cls, value):
        if not isinstance(value, list):
            return [value]
        return value
