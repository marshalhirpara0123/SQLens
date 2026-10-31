FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# CPU-only PyTorch first. The default PyPI wheel bundles CUDA libraries
# (multiple GB) that this app never uses: the embedding model is tiny and
# the LLM runs in the separate Ollama container, not here.
# torch is left unpinned on purpose: pin it from a working `pip freeze`
# before delivering to a client (see docker/README.md).
# hadolint ignore=DL3013
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY docker/requirements.docker.txt /tmp/requirements.docker.txt
RUN pip install -r /tmp/requirements.docker.txt

# Separate RUN layers on purpose: each is cached independently, so editing
# code never re-downloads torch or the model.
# Bake the embedding model into the image. This is the ONLY step besides
# pip that needs internet, and it happens at build time, so the running
# container never has to download anything.
# hadolint ignore=DL3059
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 app \
 && chown -R app:app /app /opt/hf
USER 1000

# From here on, refuse to touch the network for model files: if the baked
# model were missing the app fails loudly instead of silently downloading.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

# urlopen raises on a refused connection or a 4xx/5xx reply, which makes
# python exit non-zero, i.e. "unhealthy".
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# Builds the vector index on first start if it is empty, then runs the API.
CMD ["python", "scripts/docker_start.py"]