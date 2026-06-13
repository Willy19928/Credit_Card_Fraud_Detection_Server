ARG PYTHON_BASE_IMAGE=python:3.11.15-slim-trixie@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0
FROM ${PYTHON_BASE_IMAGE} AS builder

WORKDIR /install
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt


FROM ${PYTHON_BASE_IMAGE}

LABEL description="Credit Card Fraud Detection Inference Service"

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY --from=builder /deps /usr/local
COPY app.py .
COPY templates/ templates/
COPY models/ default-models/
COPY sample_transactions.json .
COPY docker-entrypoint.sh /usr/local/bin/fraud-inference-entrypoint

RUN mkdir -p /app/models \
    && chmod +x /usr/local/bin/fraud-inference-entrypoint \
    && chown -R appuser:appuser /app
USER appuser

ENV MODEL_PATH=/app/models/primary_mlp.pt
ENV PREPROCESSING_PATH=/app/models/preprocessing.joblib
ENV MODEL_MANIFEST_PATH=/app/models/model_manifest.json
ENV RUN_METADATA_PATH=/app/models/run_metadata.json

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import json, urllib.request; data=json.load(urllib.request.urlopen('http://localhost:5000/model_status')); raise SystemExit(0 if data['loaded'] else 1)"

ENTRYPOINT ["fraud-inference-entrypoint"]

# One threaded worker avoids duplicating the loaded model while allowing browser
# requests and health checks to run without blocking each other.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--worker-class", "gthread", \
     "--threads", "8", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
