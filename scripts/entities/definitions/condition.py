from typing import Any, Dict, List

from scripts.entities.definitions.definition import Definition
from scripts.entities.definitions.job import Job
from scripts.entities.definitions.worker import Worker


class Condition(Definition):
    DEFAULT_TAG: str = "Any"
    DEFAULT_CRITERIA: str = "All"
    DEFAULT_JOB = Job("Default", Worker("img2img"), Worker("RetinaFace"))

    def __init__(
        self, tag: str = DEFAULT_TAG, criteria: str = DEFAULT_CRITERIA, num: int = 9999, jobs: List[Job] = None
    ) -> None:
        self.tag = tag
        self.criteria = criteria
        self.num = num
        self.jobs = jobs if jobs is not None else [self.DEFAULT_JOB]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "criteria": self.criteria,
            "num": self.num,
            "jobs": [job.to_dict() for job in self.jobs],
        }

    def from_dict(self, value: Dict[str, Any]) -> "Condition":
        self.tag = value.get("tag", self.DEFAULT_TAG)
        self.criteria = value.get("criteria", self.DEFAULT_CRITERIA)
        self.num = value.get("num", 9999)

        jobs_data = value.get("jobs", [self.DEFAULT_JOB.to_dict()])
        self.jobs = [Job().from_dict(job_data) for job_data in jobs_data]
        return self
