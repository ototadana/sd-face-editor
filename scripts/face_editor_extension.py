import modules.scripts as scripts
from modules import shared

from face_editor.entities.option import Option
from face_editor.ui.ui_builder import UiBuilder
from face_editor.use_cases.image_processor import ImageProcessor
from face_editor.use_cases.workflow_manager import WorkflowManager


class FaceEditorExtension(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.__is_running = False

    def title(self):
        return "Face Editor EX"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        builder = UiBuilder(True)
        components = builder.build(is_img2img)
        self.infotext_fields = builder.infotext_fields
        return components

    def before_process_batch(self, p, enabled: bool, *args, **kwargs):
        if not enabled or self.__is_running:
            return
        option = Option(*args)
        if not option.save_original_image:
            p.do_not_save_samples = True

        if p.scripts is not None and hasattr(p.scripts, "alwayson_scripts"):
            script_index = shared.opts.data.get("face_editor_script_index", 99)
            for i, e in enumerate(p.scripts.alwayson_scripts):
                if e == self:
                    p.scripts.alwayson_scripts.insert(script_index, p.scripts.alwayson_scripts.pop(i))
                    break

    def postprocess(self, o, res, enabled, *args):
        if not enabled or self.__is_running:
            return

        option = Option(*args)
        if isinstance(enabled, dict):
            option.update_by_dict(enabled)

        try:
            self.__is_running = True

            o.do_not_save_samples = False
            ImageProcessor(WorkflowManager.get(option.workflow)).proc_images(o, res, option)

        finally:
            self.__is_running = False
