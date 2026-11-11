# Traffic Analyst: the public demo image (Google Cloud Run: python -m deploy.cloudrun; any Docker host works).
#   docker build -t traffic-analyst .
#   docker run -p 7860:7860 -e GROQ_API_KEY=... traffic-analyst      -> http://localhost:7860
# Everything a visitor needs is baked in at build time: the trained RL agent and queue predictor
# (rl/models, predict/models), the public-domain documents, their search index and both search
# models. Only the Groq key comes from outside, as a secret (environment variable), never a file.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    ANONYMIZED_TELEMETRY=False \
    TRAFFIC_OVERLAY=deploy/public.yaml \
    PORT=7860

# libgomp1: LightGBM's OpenMP runtime. curl: document download and the health check.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user (uid 1000) that owns the files it writes
# (run logs, the index, the analyst log).
RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown user /app

# Dependencies first, so a code change doesn't reinstall them (Docker layer cache).
COPY --chown=user deploy/requirements.txt deploy/requirements.txt
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r deploy/requirements.txt

USER user
COPY --chown=user . .

# The knowledge base: public-domain PDFs -> chunks -> embeddings in chroma/ (~1 min on CPU).
RUN sh deploy/fetch_docs.sh \
    && python -m deploy.prefetch \
    && python -m rag.build --docs-only

EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=5s --start-period=60s CMD curl -fs http://localhost:${PORT}/api/health || exit 1
CMD ["sh", "-c", "python -m server --host 0.0.0.0 --port ${PORT}"]
