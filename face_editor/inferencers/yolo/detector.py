from typing import List

import modules.shared as shared
from face_editor.entities.rect import Rect
from face_editor.inferencers.yolo.inferencer import YoloInferencer
from face_editor.use_cases.face_detector import FaceDetector
from PIL import Image


class YoloDetector(FaceDetector, YoloInferencer):
    def name(self) -> str:
        return "YOLO"

    def detect_faces(
        self,
        image: Image,
        path: str = "yolov8n.pt",
        repo_id: str = None,
        filename: str = None,
        conf: float = 0.5,
        **kwargs,
    ) -> List[Rect]:
        self.load_model(path, repo_id, filename)

        names = self.model.names
        output = self.model.predict(image, device=shared.device)

        faces = []
        for detection in output:
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
