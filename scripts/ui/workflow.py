import json
import os
from typing import List

import gradio as gr

from scripts.io.util import workflows_dir


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


def save_workflow(name: str, workflow: str) -> str:
    with open(os.path.join(workflows_dir, name + ".json"), "w") as file:
        file.write(workflow)
    return f"Saved to {name}.json"


def get_files() -> List[str]:
    return [f.split(".")[0] for f in os.listdir(workflows_dir) if f.endswith(".json")]


def refresh_files() -> dict:
    return gr.update(choices=get_files())


def check_json(workflow: str) -> str:
    try:
        json.loads(workflow)
        return "No errors found in the JSON."
    except json.JSONDecodeError as e:
        return f"Error in JSON: {str(e)}"


def build_workflow_ui():
    with gr.Blocks(title="Workflow"):
        with gr.Row():
            filename_dropdown = gr.Dropdown(
                choices=get_files(),
                label="Choose a Workflow",
                value="default",
                scale=3,
                multiselect=False,
                show_label=False,
            )
            refresh_button = gr.Button(value="Refresh", scale=1, size="sm")
            filename_input = gr.Textbox(scale=3, show_label=False, placeholder="Save as")
            save_button = gr.Button(value="Save", scale=1, size="sm")

        workflow_editor = gr.Code(language="json", label="Workflow", value=load_workflow("default"))
        json_status = gr.Textbox(show_label=False)

        filename_dropdown.input(load_workflow, inputs=[filename_dropdown], outputs=[workflow_editor])
        filename_dropdown.input(get_filename, inputs=[filename_dropdown], outputs=[filename_input])
        workflow_editor.change(check_json, inputs=[workflow_editor], outputs=[json_status])
        save_button.click(save_workflow, inputs=[filename_input, workflow_editor])
        refresh_button.click(refresh_files, outputs=filename_dropdown)

        return workflow_editor
