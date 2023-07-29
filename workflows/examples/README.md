# Example Workflows

- This folder contains workflow definitions that can be used as a reference when you create your own workflows.
- To access these example workflow definitions from the workflow list in the Workflow Editor, the "Search workflows in subdirectories" option must be enabled. This option is located in the Face Editor section of the "Settings" tab.
- These workflow definitions can be used as they are, or you can customize them and save them under a different name for personal use.
- Please note that some workflows require specific "Additional components" to be enabled in the Face Editor section of the "Settings" tab for them to function correctly.

---

### Example 1: Basic Workflow - MediaPipe

This workflow uses the MediaPipe face detector and applies the 'img2img' face processor and 'MediaPipe' mask generator to all detected faces.

[View the workflow definition](mediapipe.json)

Please note that to use this workflow, the 'mediapipe' option under "Additional components" in the Face Editor section of the "Settings" tab needs to be enabled.

This is a good starting point for creating more complex workflows. You can customize this by changing the face detector or face processor, adding parameters, or adding conditions to apply different processing to different faces.

---

### Example 2: Basic Workflow - YOLO Example

This workflow uses the YOLO face detector and applies the 'img2img' face processor and 'YOLO' mask generator to all detected faces. 

[View the workflow definition](yolov8n.json)

Please note that to use this workflow, the 'yolo' option under "Additional components" in the Face Editor section of the "Settings" tab needs to be enabled. Also, you need to use a model trained for face detection, as the `yolov8n.pt` model specified here does not support face detection.

Like the MediaPipe workflow, this is a good starting point for creating more complex workflows. You can customize it in the same ways.

---

### Example 3: High Accuracy Face Detection Workflow - Bingsu/adetailer Example

This workflow uses the YOLO face detector, but it employs a model that is actually capable of face detection. From our testing, the accuracy of this model in detecting faces is outstanding. For more details on the model, please check [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer).

[View the workflow definition](adetailer.json)

Please note that to use this workflow, the 'yolo' option under "Additional components" in the Face Editor section of the "Settings" tab needs to be enabled. 

---

### Example 4: Anime Face Detection Workflow - lbpcascade_animeface Example

This workflow uses the `lbpcascade_animeface` face detector, which is specially designed for detecting anime faces. The source of this detector is the widely known [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) model. 

[View the workflow definition](lbpcascade_animeface.json)

---

### Example 5: Simple Face Blurring

This workflow is straightforward and has a single, simple task - blurring detected faces in an image. It uses the `RetinaFace` method for face detection, which is a reliable and high-performance detection algorithm.

[View the workflow definition](blur.json)

The `Blur` face processor is employed here, which, as the name suggests, applies a blur effect to the detected faces. For masking, the workflow uses the `NoMask` mask generator. This is a special mask generator that doesn't mask anything - it simply allows the entire face to pass through to the face processor.

As a result, the entire area of each detected face gets blurred. This can be useful in situations where you need to anonymize faces in images for privacy reasons.

---

### Example 6: Different Processing Based on Face Position

This workflow employs the `RetinaFace` face detector and applies different processing depending on the position of the detected faces in the image. 

[View the workflow definition](blur_non_center_faces.json)

For faces located in the center of the image (as specified by the `"criteria": "center"` condition), the `img2img` face processor and `BiSeNet` mask generator are used. This means that faces in the center of the image will be subject to advanced masking and img2img transformations.

On the other hand, for faces not located in the center, the `Blur` face processor and `NoMask` mask generator are applied, effectively blurring these faces. 

This workflow could be handy in situations where you want to emphasize the subject in the middle of the photo, or to anonymize faces in the background for privacy reasons.

---

### Example 7: Basic Workflow - InsightFace

This workflow utilizes the InsightFace face detector and applies the 'img2img' face processor and 'InsightFace' mask generator to all detected faces.

[View the workflow definition](insightface.json)

Please note that to use this workflow, the 'insightface' option under "Additional components" in the Face Editor section of the "Settings" tab needs to be enabled.

---

### Example 8: Blurring Young People's Faces

This workflow employs the InsightFace face detector and adjusts the processing of detected faces based on their ages.

[View the workflow definition](blur_young_people.json)

For faces that are under 30 years old (as specified by the `"tag": "face?age<30"` condition), the Blur face processor and NoMask mask generator are applied. This results in blurring the faces of younger individuals.

For all other faces (i.e., those aged 30 or over), the img2img face processor and BiSeNet mask generator are used. This leads to advanced masking and img2img transformations.

This workflow can be beneficial in scenarios where you need to blur the faces of young people for anonymization in image processing. It also serves as an example of applying different processing based on age.

Please note that the accuracy of age detection depends on the face detector's performance. There might be variations or inaccuracies in the detected age.

