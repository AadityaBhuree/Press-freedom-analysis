# ==============================================================================
# Press Freedom Geopolitical Classifier - Production Streamlit Dockerfile
# Hardened Python 3.10-slim container with Streamlit Dashboard & ML Inference
# ==============================================================================

FROM python:3.10-slim

WORKDIR /app

# System dependencies & health check curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create unprivileged service user
RUN groupadd -g 10004 streamlitgroup && \
    useradd -u 10004 -g streamlitgroup -s /bin/bash -m streamlituser && \
    mkdir -p /app/models /app/plots && \
    chown -R streamlituser:streamlitgroup /app

# Copy application artifacts
COPY --chown=streamlituser:streamlitgroup app.py /app/app.py
COPY --chown=streamlituser:streamlitgroup preprocessing.py /app/preprocessing.py
COPY --chown=streamlituser:streamlitgroup dataset.csv /app/dataset.csv
COPY --chown=streamlituser:streamlitgroup models /app/models
COPY --chown=streamlituser:streamlitgroup plots /app/plots

USER streamlituser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
