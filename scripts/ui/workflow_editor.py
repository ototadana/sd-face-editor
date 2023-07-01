import json
import os
from typing import List

import gradio as gr
from modules import shared
from pydantic import ValidationError

from scripts.io.util import workflows_dir
from scripts.use_cases.workflow_manager import WorkflowManager


def load_workflow(file: str) -> str:
    if file is not None:
        filepath = os.path.join(workflows_dir, file + ".json")
        if os.path.isfile(filepath):
            return open(filepath).read()
    return ""


def get_filename(file: str) -> str:
    if file == "default":
        return ""
    return file


def sync_selection(file: str) -> str:
    return file


def save_workflow(name: str, workflow: str) -> str:
    if name is None or len(name) == 0:
        return ""

    with open(os.path.join(workflows_dir, name + ".json"), "w") as file:
        file.write(workflow)
    return f"Saved to {name}.json"


def get_files() -> List[str]:
    search_subdirectories = shared.opts.data.get("face_editor_search_subdirectories", False)
    files = []
    for root, _, filenames in os.walk(workflows_dir):
        if not search_subdirectories and not os.path.samefile(root, workflows_dir):
            continue
        for filename in filenames:
            if filename.endswith(".json"):
                relative_path, _ = os.path.splitext(os.path.relpath(os.path.join(root, filename), workflows_dir))
                files.append(relative_path)
    return files


def refresh_files() -> dict:
    return gr.update(choices=get_files())


def validate_workflow(workflow: str) -> str:
    try:
        json.loads(workflow)
        WorkflowManager.get(workflow)
        return "No errors found in the Workflow."
    except json.JSONDecodeError as e:
        return f"Error in JSON: {str(e)}"
    except ValidationError as e:
        errors = e.errors()
        if len(errors) == 0:
            return f"{str(e)}"
        err = errors[-1]
        return f"{' -> '.join(str(er) for er in err['loc'])} {err['msg']}\n--\n{str(e)}"
    except Exception as e:
        return f"{str(e)}"


def build(workflow_selector: gr.Dropdown):
    with gr.Blocks(title="Workflow"):
        with gr.Row():
            filename_dropdown = gr.Dropdown(
                choices=get_files(),
                label="Choose a Workflow",
                value="default",
                scale=2,
                min_width=400,
                show_label=False,
            )
            refresh_button = gr.Button(value="🔄", scale=0, size="sm", elem_classes="tool")
        with gr.Row():
            filename_input = gr.Textbox(scale=2, show_label=False, placeholder="Save as")
            save_button = gr.Button(value="💾", scale=0, size="sm", elem_classes="tool")

        workflow_editor = gr.Code(language="json", label="Workflow", value=load_workflow("default"))
        with gr.Row():
            json_status = gr.Textbox(scale=2, show_label=False)
            validate_button = gr.Button(value="✅", scale=0, size="sm", elem_classes="tool")

        filename_dropdown.input(load_workflow, inputs=[filename_dropdown], outputs=[workflow_editor])
        filename_dropdown.input(get_filename, inputs=[filename_dropdown], outputs=[filename_input])
        filename_dropdown.input(sync_selection, inputs=[filename_dropdown], outputs=[workflow_selector])

        workflow_selector.input(load_workflow, inputs=[workflow_selector], outputs=[workflow_editor])
        workflow_selector.input(get_filename, inputs=[workflow_selector], outputs=[filename_input])
        workflow_selector.input(sync_selection, inputs=[workflow_selector], outputs=[filename_dropdown])

        save_button.click(save_workflow, inputs=[filename_input, workflow_editor])

        refresh_button.click(refresh_files, outputs=[filename_dropdown])
        refresh_button.click(refresh_files, outputs=[workflow_selector])

        save_button.click(validate_workflow, inputs=[workflow_editor], outputs=[json_status])
        validate_button.click(validate_workflow, inputs=[workflow_editor], outputs=[json_status])

        return workflow_editor
