# ---------------------------------------------------------------------------
# Build stage — resolve dependencies into a virtualenv we copy wholesale
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# The pip cache is a BuildKit cache mount, not a layer: it survives an
# interrupted build (so a retry resumes instead of re-downloading every wheel)
# without adding anything to the final image.
#
# Install CPU-only torch first from the PyTorch index: the default PyPI wheel
# drags in the CUDA runtime and adds roughly 2 GB to the image.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0

WORKDIR /app
COPY requirements.txt .
# Keep torch on the CPU wheel: requirements.txt pins torch==2.14.0, which PyPI
# would otherwise resolve to the CUDA build on Linux.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/app/.cache/huggingface \
    PORT=8000

# libgomp1 is required by torch and onnxruntime; curl backs the healthcheck.
#
# /data is the mount point for the named volume holding the parent document
# store, BM25 index, and ingestion manifest. It must exist and be owned by the
# unprivileged user *before* it is ever mounted: Docker seeds a fresh named
# volume from the image content at that path, including ownership. Without this
# the volume is created root-owned and the app cannot write to it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data \
    && chown app:app /data

COPY --from=builder /opt/venv /opt/venv

USER app

# Bake the embedding model into the image so startup and the first query do not
# depend on network access to Hugging Face.
#
# Deliberately ordered before COPY src: the model download costs ~90s, and any
# source edit would otherwise invalidate this layer and re-download it on every
# rebuild.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

WORKDIR /app
COPY --chown=app:app src ./src
COPY --chown=app:app scripts ./scripts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health/ready" || exit 1

CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-2}"]
