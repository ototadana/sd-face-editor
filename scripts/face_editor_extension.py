import gradio as gr
import modules.scripts as scripts

from scripts import face_editor


class FaceEditorExtension(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.__is_running = False

    def title(self):
        return "Face Editor EX"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Face Editor", open=False, elem_id="sd-face-editor-extension"):
            script = face_editor.Script()
            enabled = gr.Checkbox(label="Enabled", value=False)
            components = [enabled] + script.ui(is_img2img)
            self.infotext_fields = [(enabled, face_editor.Option.add_prefix("enabled"))] + script.components
            return components

    def before_process_batch(self, p, enabled: bool, *args, **kwargs):
        if not enabled or self.__is_running:
            return
        option = face_editor.Option(*args)
        if not option.save_original_image:
            p.do_not_save_samples = True

        if p.scripts is not None and hasattr(p.scripts, 'alwayson_scripts') and p.scripts.alwayson_scripts[-1] != self:
            for i, e in enumerate(p.scripts.alwayson_scripts):
                if e == self:
                    p.scripts.alwayson_scripts.append(p.scripts.alwayson_scripts.pop(i))
                    break

    def postprocess(self, o, res, enabled, *args):
        if not enabled or self.__is_running:
            return

        option = face_editor.Option(*args)
        if isinstance(enabled, dict):
            option.update_by(enabled)

        try:
            self.__is_running = True

            o.do_not_save_samples = False
            script = face_editor.Script()
            mask_model, detection_model = script.get_face_models()

            script.proc_images(mask_model, detection_model, o, res, option)

        finally:
            self.__is_running = False
