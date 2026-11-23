# One image, two entry points:
#   docker run IMAGE serve   -> FastAPI on $PORT (default 8080); this is what Cloud Run runs
#   docker run IMAGE train   -> the full training pipeline (GCS download, features, training, MLflow)
#   docker run IMAGE test    -> the pytest suite
# The trained model bundle in artifacts/model/ is copied into the image, so the API starts without
# cloud credentials. Train first (python -m src.train), then build.
FROM python:3.12-slim

ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet \
    PYTHONPATH=/app \
    PORT=8080

# XGBoost needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY config.yaml ./
COPY src/ src/
COPY data_prep/ data_prep/
COPY api/ api/
COPY tests/ tests/
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY artifacts/model/ artifacts/model/

RUN useradd --create-home app && chown -R app /app && chmod +x /usr/local/bin/entrypoint.sh
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
ENTRYPOINT ["entrypoint.sh"]
CMD ["serve"]
