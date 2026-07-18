# Deploying the MC-DFM LLM interface on Streamlit Community Cloud (free)

This publishes the app (`streamlit_app.py`) as a public website on
[Streamlit Community Cloud](https://share.streamlit.io) — free, permanent, and
requiring no Docker or paid plan. Anyone can use it from a link.

## Files used

- `streamlit_app.py` — the Streamlit app (repo root, used as the "main file").
- `requirements.txt` — dependencies for the deployment (CPU torch + streamlit,
  no gradio, so no pydantic conflict).
- The generated script runs in an isolated subprocess with a 120 s timeout.

## Deploy steps

1. Make sure the repository is on GitHub (it is) and the deployment files are
   pushed to the `main` branch.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `pozzo-research-group/MC-DFM`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **Advanced settings → Python version: 3.12** (the pinned wheels in
     `requirements.txt` target Python 3.12; a different version can cause a
     missing-wheel install failure).
5. Click **Deploy**. Streamlit installs `requirements.txt` (first build takes a
   few minutes for torch) and serves the app at a public URL like
   `https://<name>.streamlit.app`. Share that link.

### If the build fails with `ModuleNotFoundError`

This means the dependency install did not complete. Common causes and fixes:

- **torch pulled the huge CUDA wheel** and timed out — `requirements.txt` pins
  `torch==2.8.0+cpu` to force the small CPU wheel, which avoids this.
- **Wrong Python version** — set Python to **3.12** in the app's Advanced
  settings (or Manage app → Settings), then reboot the app.
- Open **Manage app → Logs** to see which package failed, and adjust its pin.

## How users interact with it

1. Each user pastes **their own** AtomGPT API key (from atomgpt.org) — no shared
   secret is stored in the app.
2. They pick a model (default `gemma-4-26b`).
3. They describe a structure, click **Generate script**, then **Run generated
   script** to see the scattering curves and structure plots.

## Notes and caveats

- **Resources:** Community Cloud apps get roughly 2.7 GB RAM, enough for CPU
  torch on modest simulations. Very large `n_pairwise` or big protein systems may
  be slow or hit the memory limit.
- **Sleeping:** the app sleeps after a period of inactivity and wakes on the next
  visit (a short cold start).
- **Ephemeral storage:** files written to `sas_llm_results/` are lost when the app
  restarts. That is fine for interactive use (results are shown in the page); it
  just means outputs are not persisted server-side.
- **Executing generated code:** the app runs LLM-generated Python in a subprocess
  with a timeout. That keeps it out of the server process and bounds runtime, but
  a subprocess on shared infrastructure is not a hard security boundary. Keep the
  app scoped to trusted/experimental use, or add stronger sandboxing for a widely
  shared public link.
- **Data files:** the geometric-shape demos (sphere, cylinder, pyramid) need no
  PDB files. Protein examples that load from `Data/PDB/` will only work if those
  files are present in the repo the app is deployed from.
