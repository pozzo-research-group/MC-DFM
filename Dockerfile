# Dockerfile for a Hugging Face Docker Space (free CPU tier).
# Runs the MC-DFM Gradio app defined in sas_llm/gradio_app.py via app.py.

FROM python:3.12-slim

# Hugging Face Docker Spaces run as a non-root user with uid 1000.
RUN useradd -m -u 1000 user

WORKDIR /app

# Install Python dependencies first for better layer caching.
# requirements-hf.txt is the gradio-based dependency set for this Space
# (requirements.txt at the repo root is for the Streamlit deployment).
COPY --chown=user requirements-hf.txt .
RUN pip install --no-cache-dir -r requirements-hf.txt

# Copy the rest of the repository.
COPY --chown=user . .

USER user
ENV HOME=/home/user \
    MCDFM_SANDBOX=1 \
    PORT=7860

# Hugging Face Docker Spaces expect the app on port 7860.
EXPOSE 7860

CMD ["python", "app.py"]
