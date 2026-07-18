"""Hugging Face Spaces entry point for the MC-DFM LLM interface.

Works for both a Gradio SDK Space and a Docker SDK Space. This wrapper enables
subprocess sandboxing of the LLM-generated code (so it never runs in the server
process) and launches the Gradio app defined in ``sas_llm/gradio_app.py`` on the
port Hugging Face expects (7860).
"""

import os

# Run LLM-generated scripts in an isolated subprocess with a timeout.
os.environ.setdefault("MCDFM_SANDBOX", "1")

from sas_llm.gradio_app import demo

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
