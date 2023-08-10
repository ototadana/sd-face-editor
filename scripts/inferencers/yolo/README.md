# YOLO components
Component implementation using [YOLO](https://github.com/ultralytics/ultralytics).

To use this, please enable 'yolo' option under "Additional components" in the Face Editor section of the "Settings" tab. 

## 1. Face Detector
This component utilizes an object detection model from YOLO for face detection. Though not exclusively designed for face detection, it can identify other objects of interest as well. Effectiveness can be enhanced by utilizing a model specifically trained on faces.

#### Name
- YOLO

#### Implementation
- [YoloDetector](detector.py)

#### Recognized UI settings
- N/A

#### Configuration Parameters (in JSON)
- `path` (string, default: "yolov8n.pt"): Path to the model file. If `repo_id` is specified, the model will be downloaded from Hugging Face Model Hub instead, using `repo_id` and `filename`.
- `repo_id` (string, optional): The repository ID if the model is hosted on Hugging Face Model Hub. If this is specified, `path` will be ignored.
- `filename` (string, optional): The filename of the model in the Hugging Face Model Hub repository. Use this in combination with `repo_id`.
- `conf`: (float, optional, default: 0.5): The confidence threshold for object detection.

#### Returns
- tag: The class of the detected object (as trained in the YOLO model)
- attributes: N/A
- landmarks: N/A

#### Usage in Workflows
- [yolov8n.json](../../../workflows/examples/yolov8n.json)
- [adetailer.json](../../../workflows/examples/adetailer.json)


---

## 2. Mask Generator
This component utilizes a segmentation model from YOLO (You Only Look Once) for mask generation.

#### Name
- YOLO

#### Implementation
- [YoloMaskGenerator](mask_generator.py)

#### Recognized UI settings
- Use minimal area (for close faces)

#### Configuration Parameters (in JSON)
- `path` (string, default: "yolov8n-seg.pt"): Path to the model file. If `repo_id` is specified, the model will be downloaded from Hugging Face Model Hub instead, using `repo_id` and `filename`.
- `repo_id` (string, optional): The repository ID if the model is hosted on Hugging Face Model Hub. If this is specified, `path` will be ignored.
- `filename` (string, optional): The filename of the model in the Hugging Face Model Hub repository. Use this in combination with `repo_id`.
- `conf` (float, default: 0.5): Confidence threshold for detections. Any detection with a confidence lower than this will be ignored.

#### Usage in Workflows
- [yolov8n.json](../../../workflows/examples/yolov8n.json)
