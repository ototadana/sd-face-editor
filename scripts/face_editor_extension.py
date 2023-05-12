import gradio as gr
import modules.scripts as scripts

from scripts import face_editor


class FaceEditorExtension(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.__is_running = False

    def title(self):
        return "Face Editor"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Face Editor", open=False, elem_id="sd-face-editor-extension"):
            return [gr.Checkbox(label="Enabled", value=False)] + face_editor.Script().ui(is_img2img)

    def before_process_batch(self, p, enabled: bool,
                             face_margin: float,
                             confidence: float,
                             strength1: float,
                             strength2: float,
                             max_face_count: int,
                             mask_size: int,
                             mask_blur: int,
                             prompt_for_face: str,
                             apply_inside_mask_only: bool,
                             save_original_image: bool,
                             show_intermediate_steps: bool,
                             apply_scripts_to_faces: bool, **kwargs):
        if not enabled or self.__is_running:
            return
        if not save_original_image:
            p.do_not_save_samples = True

    def postprocess(self, o, res,
                    enabled: bool,
                    face_margin: float,
                    confidence: float,
                    strength1: float,
                    strength2: float,
                    max_face_count: int,
                    mask_size: int,
                    mask_blur: int,
                    prompt_for_face: str,
                    apply_inside_mask_only: bool,
                    save_original_image: bool,
                    show_intermediate_steps: bool,
                    apply_scripts_to_faces: bool):
        if not enabled or self.__is_running:
            return

        o.do_not_save_samples = False
        script = face_editor.Script()
        mask_model, detection_model = script.get_face_models()

        try:
            self.__is_running = True
            script.proc_images(mask_model, detection_model, o, res,
                               face_margin=face_margin, confidence=confidence,
                               strength1=strength1, strength2=strength2,
                               max_face_count=max_face_count, mask_size=mask_size,
                               mask_blur=mask_blur, prompt_for_face=prompt_for_face,
                               apply_inside_mask_only=apply_inside_mask_only,
                               apply_scripts_to_faces=apply_scripts_to_faces,
                               )

        finally:
            self.__is_running = False
