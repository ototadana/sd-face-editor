import modules.scripts as scripts
import modules.shared as shared
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img, process_images

from face_editor.entities.option import Option
from face_editor.ui.ui_builder import UiBuilder
from face_editor.use_cases.image_processor import ImageProcessor
from face_editor.use_cases.workflow_manager import WorkflowManager


class Script(scripts.Script):
    def title(self):
        return "Face Editor"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        builder = UiBuilder(False)
        components = builder.build(is_img2img)
        self.infotext_fields = builder.infotext_fields
        return components

    def run(self, o: StableDiffusionProcessing, *args):
        option = Option(*args)
        processor = ImageProcessor(WorkflowManager.get(option.workflow))

        if (
            isinstance(o, StableDiffusionProcessingImg2Img)
            and o.n_iter == 1
            and o.batch_size == 1
            and not option.apply_scripts_to_faces
        ):
            return processor.proc_image(o, option)
        else:
            shared.state.job_count = o.n_iter * 3
            if not option.save_original_image:
                o.do_not_save_samples = True
            res = process_images(o)
            o.do_not_save_samples = False

            return processor.proc_images(o, res, option)
