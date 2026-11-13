FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install runtime dependencies (Tesseract OCR, headless OpenCV/PyMuPDF libs, curl)
# Models are mounted via runtime volumes; nothing is downloaded during build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root execution user and directories
RUN groupadd --gid 10001 documind \
    && useradd --uid 10001 --gid documind --create-home --shell /bin/bash documind \
    && mkdir -p /app/storage/uploads /app/checkpoints /app/data/feedback_exports \
    && chown -R documind:documind /app

COPY --chown=documind:documind backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

COPY --chown=documind:documind backend /app/backend
COPY --chown=documind:documind ml /app/ml
COPY --chown=documind:documind backend/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

USER documind

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

