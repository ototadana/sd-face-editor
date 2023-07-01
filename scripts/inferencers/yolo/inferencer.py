from typing import Optional

from huggingface_hub import hf_hub_download
from ultralytics import YOLO


class YoloInferencer:
    def __init__(self, task: str = None):
        self.task = task
        self.model = None
        self.loaded_path: Optional[str] = None
        self.loaded_repo_id: Optional[str] = None
        self.loaded_file_name: Optional[str] = None

    def load_model(self, path: str = None, repo_id: str = None, filename: str = None):
        if self.model is None or path != self.loaded_path or repo_id != self.loaded_repo_id:
            if repo_id is not None:
                path = hf_hub_download(repo_id, filename)
            self.model = YOLO(path, task=self.task)
            self.loaded_path = path
            self.loaded_repo_id = repo_id
            self.loaded_file_name = filename
