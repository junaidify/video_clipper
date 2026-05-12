# ── Video Auto-Clipper ──
# Multi-stage Docker build for Railway / Render / any Docker host

FROM python:3.11-slim AS base

# System deps: FFmpeg, fonts, and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    fonts-liberation \
    fontconfig \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Rebuild font cache so FFmpeg drawtext can find fonts
RUN fc-cache -fv

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p uploads clips_output video_library training_sessions

# Railway injects PORT env var; default to 5000
ENV PORT=5000
ENV FLASK_DEBUG=false
ENV PYTHONUNBUFFERED=1

EXPOSE ${PORT}

# Use gunicorn for production (handles concurrency properly)
RUN pip install --no-cache-dir gunicorn

CMD gunicorn --bind 0.0.0.0:${PORT} \
    --workers 2 \
    --threads 4 \
    --timeout 300 \
    --max-requests 100 \
    --max-requests-jitter 20 \
    app:app
