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
import asyncio
import subprocess

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sas_llm.atomgpt_llm import use_llm, list_atomgpt_models

DEFAULT_SAVE_DIR = "sas_llm_results/"
DEFAULT_MODEL = "gemma-4-26b"
RUN_TIMEOUT_SECONDS = 120
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_NOTEBOOKS_DIR = os.path.join(_REPO_ROOT, "Notebooks")

st.set_page_config(page_title="MC-DFM LLM Interface", page_icon="🔬")
st.title("MC-DFM LLM Interface")
st.markdown(
    "Describe a structure in plain language and the AtomGPT LLM will generate "
    "an MC-DFM Python script to simulate its scattering curve. "
    "Create a free account at [atomgpt.org](https://atomgpt.org) and copy your "
    "API key from **Settings → Account → Show API key**."
)

api_key = st.text_input("AtomGPT API Key", type="password")


@st.cache_data(show_spinner=False)
def get_models(key):
    """Fetches the list of available AtomGPT models (cached per API key)."""
    return list_atomgpt_models(key)


# Model dropdown, populated from the models currently available on AtomGPT once
# an API key is entered. Falls back to the default model until then.
if api_key:
    available_models = get_models(api_key) or [DEFAULT_MODEL]
else:
    available_models = [DEFAULT_MODEL]

default_index = available_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in available_models else 0
model = st.selectbox("Model", available_models, index=default_index)

save_dir = DEFAULT_SAVE_DIR
instructions = st.text_area(
    "Instructions",
    placeholder="Simulate the scattering of a square-based pyramid with an edge length of 10 nm and a height of 10 nm.",
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
        before = set(os.listdir(save_dir))
        with st.spinner("Generating code with AtomGPT..."):
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
    st.subheader("Generated script (editable)")
    st.caption("Edit the code below before running if you want to tweak or fix it.")
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
