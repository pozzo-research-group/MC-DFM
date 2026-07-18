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
import glob
import asyncio
import subprocess

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sas_llm.atomgpt_llm import use_llm

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
model = st.text_input("Model", value=DEFAULT_MODEL)
save_dir = st.text_input("Save directory", value=DEFAULT_SAVE_DIR) or DEFAULT_SAVE_DIR
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

# --- Show the generated script and offer to run it ---
if st.session_state.code:
    st.subheader("Generated script")
    st.code(st.session_state.code, language="python")

    if st.button("Run generated script"):
        folder = st.session_state.folder
        script_path = os.path.abspath(os.path.join(folder, "generated_script.py"))
        proc = None
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
                pngs = sorted(glob.glob(os.path.join(os.path.abspath(folder), "*.png")))
                for png in pngs:
                    st.image(png)
                if not pngs:
                    st.info("The script ran but did not save any plot images.")
