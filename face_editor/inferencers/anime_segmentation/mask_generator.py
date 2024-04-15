from typing import Tuple

import cv2
import huggingface_hub
import modules.shared as shared
import numpy as np
import onnxruntime as rt
from face_editor.use_cases.mask_generator import MaskGenerator


class AnimeSegmentationMaskGenerator(MaskGenerator):
    def name(self) -> str:
        return "AnimeSegmentation"

    def generate_mask(
        self,
        face_image: np.ndarray,
        face_area_on_image: Tuple[int, int, int, int],
        **kwargs,
    ) -> np.ndarray:
        device_id = shared.cmd_opts.device_id if shared.cmd_opts.device_id is not None else 0
        model_path = huggingface_hub.hf_hub_download("skytnt/anime-seg", "isnetis.onnx")
        model = rt.InferenceSession(
            model_path,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": device_id,
                    },
                ),
                "CPUExecutionProvider",
            ],
        )

        mask = self.__get_mask(face_image, model)
        mask = (mask * 255).astype(np.uint8)
        return mask.repeat(3, axis=2)

    def __get_mask(self, face_image, model):
        face_image = (face_image / 255).astype(np.float32)
        image_input = cv2.resize(face_image, (1024, 1024))
        image_input = np.transpose(image_input, (2, 0, 1))
        image_input = image_input[np.newaxis, :]
        mask = model.run(None, {"img": image_input})[0][0]
        mask = np.transpose(mask, (1, 2, 0))
        mask = cv2.resize(mask, (face_image.shape[0], face_image.shape[1]))[:, :, np.newaxis]
        return mask
