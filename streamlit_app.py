"""Streamlit interface for the AtomGPT MC-DFM workflow.

Designed to be deployed for free on Streamlit Community Cloud
(https://share.streamlit.io): point it at this repository, branch ``main``,
with ``streamlit_app.py`` as the main file. Users enter their own AtomGPT API
key, describe a structure, generate the MC-DFM script, and run it to see the
scattering curves and structure plots.

The generated script is executed in an isolated subprocess with a timeout so
LLM-generated code does not run in the server process.
"""

import matplotlib
# Set a non-interactive backend before anything imports pyplot.
matplotlib.use("Agg", force=True)

import os
import sys
import time
import glob
import shutil
import asyncio
import subprocess
from datetime import datetime

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sas_llm.atomgpt_llm import use_llm, list_atomgpt_models

DEFAULT_SAVE_DIR = "sas_llm_results/"
DEFAULT_MODEL = "gemma-4-26b"
RUN_TIMEOUT_SECONDS = 120
MAX_SAVED_RUNS = 25          # keep only the most recent N result folders on disk
USER_LOG_FILE = "user_inputs_log.txt"
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_NOTEBOOKS_DIR = os.path.join(_REPO_ROOT, "Notebooks")


def log_user_input(save_dir, model, instructions):
    """Records a user's prompt so the app owner can see what is being run.

    The line is printed to stdout (visible only to the owner in the Streamlit
    Cloud "Manage app -> Logs" panel) and appended to a central log file that is
    exempt from pruning. Note: both are on the ephemeral filesystem and reset
    when the app restarts; use an external store for durable logging.
    """
    line = f"{datetime.now().isoformat(timespec='seconds')}\tmodel={model}\t{instructions!r}"
    print("USER_INPUT\t" + line, flush=True)
    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, USER_LOG_FILE), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def prune_old_runs(save_dir, keep=MAX_SAVED_RUNS):
    """Deletes all but the most recent ``keep`` result folders to bound disk use.

    Only directories are pruned; the central user-input log file is left intact.
    """
    if not os.path.isdir(save_dir):
        return
    folders = [
        os.path.join(save_dir, d) for d in os.listdir(save_dir)
        if os.path.isdir(os.path.join(save_dir, d))
    ]
    folders.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for old in folders[keep:]:
        shutil.rmtree(old, ignore_errors=True)

st.set_page_config(page_title="MC-DFM LLM Interface", page_icon="🔬", layout="wide")


@st.cache_data(show_spinner=False)
def get_models(key):
    """Fetches the list of available AtomGPT models (cached per API key)."""
    return list_atomgpt_models(key)


# Centered header image (the project's scattering render).
_HEADER_IMG = os.path.join(_REPO_ROOT, "Images", "RhuA1.png")
_hc1, _hc2, _hc3 = st.columns([1, 2, 1])
with _hc2:
    if os.path.exists(_HEADER_IMG):
        st.image(_HEADER_IMG, use_container_width=True)

st.title("🔬 MC-DFM LLM Interface")
st.markdown(
    "Describe a structure in plain language and the LLM will generate "
    "a Python script to simulate its small angle scattering curve. "
    "This tool uses a LLM to generate the desired structure "
    "in real-space using geometric functions and then uses the MC-DFM, "
    "a python implementation of the numerical solution to the Debye Scattering "
    "Equation, to calculate the scattering curve. It works best for geometric "
    "structures (e.g., spheres, tetrahedrons, pyramids, cones) and their assemblies."
)
st.info(
    "This app is intended as a simple demonstration of the MC-DFM LLM workflow "
    "and is limited by the computational power available to it. For more "
    "sophisticated use — such as large complex assemblies or protein assemblies — "
    "download and run the code from "
    "[github.com/pozzo-research-group/MC-DFM](https://github.com/pozzo-research-group/MC-DFM)."
)

# --- Sidebar: credentials and model selection ---
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input(
        "AtomGPT API Key",
        type="password",
        help="Create a free account at atomgpt.org, then copy your key from "
             "Settings → Account → Show API key.",
    )
    available_models = (get_models(api_key) or [DEFAULT_MODEL]) if api_key else [DEFAULT_MODEL]
    default_index = available_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in available_models else 0
    model = st.selectbox("Model", available_models, index=default_index)
    st.markdown("[Get a free API key ↗](https://atomgpt.org)")

save_dir = DEFAULT_SAVE_DIR

# --- Main: describe the structure ---
st.subheader("Describe your structure")

EXAMPLES = {
    "🔵 Sphere": "Simulate the scattering of a sphere with a radius of 50 angstroms.",
    "🔺 Pyramid": "Simulate the scattering of a square-based pyramid with an edge length of 10 nm and a height of 10 nm.",
    "🧅 Core–shell": "Simulate the scattering of a core-shell sphere with a core radius of 40 angstroms and a shell thickness of 15 angstroms.",
    "⬡ Dimer": "Simulate the scattering of a dimer of spheres, each 30 angstroms in radius, separated by 80 angstroms.",
}
st.caption("Need a starting point? Click an example to fill the box:")
ex_cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(ex_cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state["instructions_input"] = text

st.session_state.setdefault("instructions_input", "")
instructions = st.text_area(
    "Instructions",
    key="instructions_input",
    placeholder="Simulate the scattering of a square-based pyramid with an edge length of 10 nm and a height of 10 nm.",
    height=120,
)

if "folder" not in st.session_state:
    st.session_state.folder = None
    st.session_state.code = None

# --- Generate the script ---
if st.button("Generate script", type="primary"):
    if not api_key:
        st.error("Please enter your AtomGPT API key.")
    elif not instructions.strip():
        st.error("Please enter instructions.")
    else:
        os.makedirs(save_dir, exist_ok=True)
        # Record the user's prompt (owner-visible) and bound disk use.
        log_user_input(save_dir, model, instructions)
        prune_old_runs(save_dir)
        before = set(os.listdir(save_dir))
        with st.spinner("Generating code ..."):
            try:
                asyncio.run(use_llm(api_key, model, instructions, save_dir))
            except Exception as e:
                st.error(f"Error calling AtomGPT: {type(e).__name__}: {e}")
        new_folders = set(os.listdir(save_dir)) - before
        if not new_folders:
            st.error("No script was generated. Check the API key, model name, and AtomGPT status.")
        else:
            folder = os.path.join(save_dir, sorted(new_folders)[-1])
            with open(os.path.join(folder, "generated_script.py"), "r", encoding="utf-8") as f:
                st.session_state.code = f.read()
            st.session_state.folder = folder
            # Seed the editable code box with the freshly generated script.
            st.session_state.editor = st.session_state.code

# --- Show the (editable) generated script and offer to run it ---
if st.session_state.code:
    st.divider()
    st.subheader("Generated script (editable)")
    st.caption("Edit the code below before running if you want to tweak or fix it. A common edit is the desired q-range.")
    st.session_state.setdefault("editor", st.session_state.code)
    edited_code = st.text_area("Python script", key="editor", height=400)

    if st.button("Run generated script"):
        folder = st.session_state.folder
        script_path = os.path.abspath(os.path.join(folder, "generated_script.py"))
        # Run whatever is currently in the editor (the user's edits included).
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(st.session_state.editor)
        proc = None
        # Files newer than this are outputs produced by this run. The buffer
        # avoids clock/rounding issues.
        run_start = time.time() - 1
        with st.spinner("Running the simulation..."):
            try:
                proc = subprocess.run(
                    [sys.executable, script_path],
                    cwd=_NOTEBOOKS_DIR,
                    capture_output=True,
                    text=True,
                    timeout=RUN_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                st.error(f"Generated script timed out after {RUN_TIMEOUT_SECONDS} s.")

        if proc is not None:
            output = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode != 0:
                st.error("Error while running generated_script.py:")
                st.code(output[-2000:])
            else:
                st.success("Script ran successfully.")
                if output.strip():
                    st.text(output[-500:])

                # Collect every file the run produced anywhere under the results
                # directory (by modification time), so outputs are found even if
                # the script saved them into a different results subfolder.
                images, data_files = [], []
                for root, _dirs, files in os.walk(save_dir):
                    for fn in files:
                        p = os.path.join(root, fn)
                        try:
                            if os.path.getmtime(p) < run_start:
                                continue
                        except OSError:
                            continue
                        if fn == USER_LOG_FILE:
                            continue
                        low = fn.lower()
                        if low.endswith(".png"):
                            images.append(p)
                        elif low.endswith((".txt", ".csv", ".npy", ".dat")):
                            data_files.append(p)

                images.sort()
                data_files.sort()

                if images:
                    st.subheader("Plots")
                    for img in images:
                        st.image(img, caption=os.path.basename(img))
                else:
                    st.info("The script ran but did not save any plot images.")

                # Download buttons for the plots and the scattering data.
                downloads = images + [d for d in data_files
                                      if os.path.basename(d) != "generated_script.py"]
                if downloads:
                    st.subheader("Download outputs")
                    for i, path in enumerate(downloads):
                        with open(path, "rb") as fh:
                            st.download_button(
                                label=f"Download {os.path.basename(path)}",
                                data=fh.read(),
                                file_name=os.path.basename(path),
                                key=f"dl_{i}",
                            )
