"""
Browser interface for the AtomGPT MC-DFM workflow.

Run from the repository root after installing the package:

    pip install .
    python -m sas_llm.gradio_app

This opens a local browser tab where users can enter their AtomGPT API key,
describe the structure they want to simulate, generate the MC-DFM script,
and execute it without touching a notebook.
"""

import matplotlib
# Must be set before any other module (including the generated scripts, which
# import Scattering_Simulator.fitting) gets a chance to import pyplot with an
# interactive GUI backend. Interactive backends like TkAgg require GUI calls
# on the main thread, but Gradio runs callbacks in worker threads, which
# crashes plotting. Agg renders to static images instead, which is also all
# Gradio's Gallery component can display anyway.
matplotlib.use("Agg", force=True)

import os
import sys
import io
import glob
import asyncio
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sas_llm.atomgpt_llm import use_llm

DEFAULT_SAVE_DIR = "sas_llm_results/"


def generate_script(api_key, model, mode, instructions, save_dir):
    if not api_key:
        return "ERROR: Please enter your AtomGPT API key.", None, gr.update(value=None)

    save_dir = save_dir or DEFAULT_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    before = set(os.listdir(save_dir))
    asyncio.run(use_llm(api_key, model, mode, instructions, save_dir))
    after = set(os.listdir(save_dir))
    new_folders = after - before

    if not new_folders:
        return "ERROR: No script was generated. Check the API key and AtomGPT status.", None, gr.update(value=None)

    folder = os.path.join(save_dir, sorted(new_folders)[-1])
    script_path = os.path.join(folder, "generated_script.py")

    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()

    return code, folder, gr.update(value=folder)


_NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Notebooks")


def run_script(folder):
    if not folder:
        return "ERROR: Generate a script first.", []

    script_path = os.path.abspath(os.path.join(folder, "generated_script.py"))
    if not os.path.exists(script_path):
        return f"ERROR: {script_path} not found.", []

    plt.close("all")

    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()

    # The generated code uses paths like '../Data/...' and '../sas_llm_results',
    # which assume the working directory is Notebooks/ (as in the LLM's prompt examples).
    original_cwd = os.getcwd()
    try:
        os.chdir(_NOTEBOOKS_DIR)
        exec(compile(code, script_path, "exec"), {"__name__": "__main__"})
    except Exception as e:
        return f"ERROR while running generated_script.py:\n{e}", []
    finally:
        os.chdir(original_cwd)

    images = []
    for n in plt.get_fignums():
        fig = plt.figure(n)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        images.append(Image.open(buf))

    return "Script ran successfully.", images


with gr.Blocks(title="MC-DFM LLM Interface") as demo:
    gr.Markdown("# MC-DFM LLM Interface")
    gr.Markdown(
        "Describe a structure in plain language and the AtomGPT LLM will generate "
        "an MC-DFM Python script to simulate its scattering curve. "
        "Create a free account at [atomgpt.org](https://atomgpt.org) and copy your "
        "API key from Settings -> Account -> Show API key."
    )

    with gr.Row():
        api_key = gr.Textbox(label="AtomGPT API Key", type="password")
        model = gr.Textbox(label="Model", value="openai/gpt-oss-20b")

    with gr.Row():
        mode = gr.Radio(
            ["weighted_sum", "distribution"],
            label="Polydispersity mode",
            value="weighted_sum",
        )
        save_dir = gr.Textbox(label="Save directory", value=DEFAULT_SAVE_DIR)

    instructions = gr.Textbox(
        label="Instructions",
        lines=4,
        placeholder="Simulate the scattering of a square-based pyramid with an edge length of 10 nm and a height of 10 nm.",
    )

    generate_btn = gr.Button("Generate script", variant="primary")
    code_output = gr.Code(label="Generated script", language="python")
    folder_state = gr.Textbox(visible=False)

    generate_btn.click(
        fn=generate_script,
        inputs=[api_key, model, mode, instructions, save_dir],
        outputs=[code_output, folder_state, folder_state],
    )

    run_btn = gr.Button("Run generated script")
    run_status = gr.Textbox(label="Status")
    run_plots = gr.Gallery(label="Plots")

    run_btn.click(
        fn=run_script,
        inputs=[folder_state],
        outputs=[run_status, run_plots],
    )


if __name__ == "__main__":
    demo.launch()
