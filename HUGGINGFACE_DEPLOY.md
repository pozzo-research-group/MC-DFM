# Deploying the MC-DFM LLM interface to Hugging Face Spaces

This guide publishes the Gradio app (`sas_llm/gradio_app.py`) as a public
Hugging Face Space so anyone can use it from a link, no terminal required.

There are two free ways to do this — pick one:

- **Gradio SDK Space** (Section A) — simplest; uses `app.py` + `requirements.txt`.
- **Docker SDK Space** (Section B) — same free CPU tier, full control; uses the
  `Dockerfile`. Use this if the Gradio SDK is unavailable for your account.

> **A "Static" Space will not work.** Static Spaces serve only HTML/CSS/JS with
> no Python backend. This app needs server-side Python to run the PyTorch-based
> MC-DFM simulation, call the AtomGPT API, and execute the generated script.
> Browser-only Python (Gradio-Lite / Pyodide) cannot load PyTorch, so the
> simulation cannot run client-side.

## Files already prepared

- `app.py` — the Space entry point (works for both Gradio and Docker Spaces).
  Enables subprocess sandboxing and launches the Gradio app on port 7860.
- `requirements.txt` — CPU dependencies for the Space (gradio 6.x + CPU torch).
- `Dockerfile` — for the Docker SDK Space route.
- The app runs LLM-generated code in an isolated subprocess with a timeout when
  `MCDFM_SANDBOX=1` (set automatically by `app.py`).

## Section A — Gradio SDK Space

1. Create a free account at https://huggingface.co.
2. Click **New → Space**.
   - **SDK:** Gradio
   - **Space hardware:** **CPU basic — Free** (enough for modest simulations).
   - Give it a name, e.g. `mc-dfm-llm`.
3. Hugging Face creates the Space with a `README.md` that contains a YAML
   metadata header. Make sure that header reads like this (edit it in the
   Space's **Files** tab or **Settings**):

   ```yaml
   ---
   title: MC-DFM LLM
   emoji: 🔬
   colorFrom: blue
   colorTo: indigo
   sdk: gradio
   sdk_version: 6.20.0
   app_file: app.py
   pinned: false
   ---
   ```

   > Keep this header only in the **Space's** README, not the GitHub README.

## Section B — Docker SDK Space (free, if Gradio SDK is unavailable)

1. Create a free account at https://huggingface.co.
2. Click **New → Space**.
   - **SDK:** Docker → **Blank** template
   - **Space hardware:** **CPU basic — Free**
   - Give it a name, e.g. `mc-dfm-llm`.
3. Set the Space's `README.md` metadata header to use the Docker SDK and port
   7860:

   ```yaml
   ---
   title: MC-DFM LLM
   emoji: 🔬
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

   The included `Dockerfile` installs `requirements.txt` and runs `app.py`.
   Everything else (the push step below, usage, caveats) is identical to the
   Gradio route.

## Push the code to the Space

The Space is its own git repo. From this repository:

```
git remote add space https://huggingface.co/spaces/<your-username>/mc-dfm-llm
git push space main
```

Hugging Face then installs `requirements.txt` and launches `app.py`. The first
build takes a few minutes (torch download). When it finishes, the Space is live
at `https://huggingface.co/spaces/<your-username>/mc-dfm-llm` — share that link.

## Keeping the Space slim (recommended)

The full repository contains large data files (e.g. `Data/PDB/8ac6.pdb` at
75 MB) that the geometric-shape demos (sphere, cylinder, pyramid) do not need —
those PDB files are only for the protein examples. To keep the Space small and
fast to build, push a branch that excludes `Data/PDB/`, large `.gsd`/`.xyz`
trajectories, and the `Images/` and `sas_llm_results/` history. The app only
needs:

- `app.py`, `requirements.txt`
- `sas_llm/`, `Scattering_Simulator/`, `genetic_algorithm/`
- an (even empty) `Notebooks/` directory — the generated scripts run with that
  as the working directory
- an (even empty) `sas_llm_results/` directory for the generated output

## How users interact with it

1. Each user pastes **their own** AtomGPT API key (from atomgpt.org) into the
   password field — no shared secret is stored in the Space.
2. They choose a model (default `gemma-4-26b`; the current list is available via
   `print_atomgpt_models` in the notebook).
3. They describe a structure, click **Generate script**, then **Run generated
   script** to see the scattering curves and structure plots.

## Notes and caveats

- **Free tier sleeps** after ~48 h of inactivity and cold-starts on the next
  visit. Upgrade the Space hardware to avoid this or to get a GPU for faster
  simulations.
- **Sandboxing:** with `MCDFM_SANDBOX=1`, generated code runs in a subprocess
  with a 120 s timeout, and plots are collected from the PNG files the script
  saves. This is safer than in-process `exec`, but a subprocess on the same
  machine is not a full security boundary — for stronger isolation, run the
  Space on dedicated hardware or add container-level sandboxing.
- **Ephemeral storage:** files written to `sas_llm_results/` on a free Space are
  lost on restart. If users need to keep outputs, add the paid persistent-storage
  add-on or have the app offer results as downloads.
