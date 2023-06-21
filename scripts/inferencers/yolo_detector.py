from typing import List

from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

from scripts.entities.rect import Rect
from scripts.use_cases.face_detector import FaceDetector


class YoloDetector(FaceDetector):
    def __init__(self):
        self.model = None
        self.loaded_path = None
        self.loaded_repo_id = None

    def name(self) -> str:
        return "YOLO"

    def detect_faces(
        self, image: Image, path: str = None, repo_id: str = None, filename: str = None, conf: float = 0.5, **kwargs
    ) -> List[Rect]:
        if self.model is None or path != self.loaded_path or repo_id != self.loaded_repo_id:
            self.__load_model(path, repo_id, filename)

        names = self.model.names
        output = self.model.predict(image)

        faces = []
        for i, detection in enumerate(output):
            boxes = detection.boxes
            for box in boxes:
                tag = names[int(box.cls)]
                box_conf = float(box.conf[0])
                if box_conf >= conf:
                    l, t, r, b = (
                        int(box.xyxy[0][0]),
                        int(box.xyxy[0][1]),
                        int(box.xyxy[0][2]),
                        int(box.xyxy[0][3]),
                    )
                    faces.append(Rect(l, t, r, b, tag))
                    print(f"{tag} detected at ({l}, {t}, {r}, {b}), Confidence: {box_conf:.2f}")

        return faces

    def __load_model(self, path: str = None, repo_id: str = None, filename: str = None):
        if repo_id is not None:
            path = hf_hub_download(repo_id, filename)
        self.model = YOLO(path)
        self.loaded_path = path
        self.loaded_repo_id = repo_id
