# ── Build stage: install heavy deps ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="your-email@example.com"
LABEL description="Fire Detection Inference Service"

# 非 root 使用者（安全最佳實踐）
RUN useradd -m -u 1000 appuser

WORKDIR /app

# 複製已安裝的套件
COPY --from=builder /deps /usr/local

# 複製應用程式
COPY app.py .
COPY templates/ templates/

# 建立模型上傳目錄，給予寫入權限
RUN mkdir -p /app/models && chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/model_status')" || exit 1

CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
