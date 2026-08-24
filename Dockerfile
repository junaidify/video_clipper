# ==============================================================================
# Dockerfile: Video Auto-Clipper & Production Studio
# Optimized for Railway.app, Render, DigitalOcean, Hetzner, and Docker VPS
# ==============================================================================

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=5000 \
    WHISPER_CACHE_DIR=/root/.cache/whisper

# Install system dependencies: FFmpeg, fonts, download accelerators, and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    aria2 \
    fonts-dejavu-core \
    fonts-liberation \
    fonts-freefont-ttf \
    fontconfig \
    ca-certificates \
    curl \
    git \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Install Python dependencies (leveraging Docker layer cache)
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Step 2: Pre-download default Whisper 'base' model during build
# This avoids 150MB+ download delays when processing the first video in production
RUN python -c "import whisper; whisper.load_model('base')"

# Step 3: Copy application source code
COPY . .

# Step 4: Create required runtime directories
RUN mkdir -p uploads clips_output video_library training_sessions

EXPOSE ${PORT}

# Healthcheck for container orchestrators (Railway, Render, Docker Swarm, K8s)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/settings || exit 1

# Startup script:
# 1. Decodes YOUTUBE_COOKIES_B64 (if provided) into /app/cookies.txt
# 2. Launches Gunicorn WSGI server with multi-threading and extended timeout
CMD bash -c '\
    if [ -n "$YOUTUBE_COOKIES_B64" ]; then \
        echo "$YOUTUBE_COOKIES_B64" | tr -d "\r\n " | base64 -d > /app/cookies.txt 2>/dev/null; \
        if [ -s /app/cookies.txt ]; then \
            SIZE=$(wc -c < /app/cookies.txt); \
            LINES=$(wc -l < /app/cookies.txt); \
            echo "[startup] Successfully decoded cookies -> /app/cookies.txt (${SIZE} bytes, ${LINES} lines)"; \
        else \
            echo "[startup] WARNING: YOUTUBE_COOKIES_B64 produced empty file"; \
            rm -f /app/cookies.txt; \
        fi; \
    else \
        echo "[startup] No YOUTUBE_COOKIES_B64 configured (standard YouTube downloads active)"; \
    fi && \
    exec gunicorn --bind 0.0.0.0:${PORT:-10000} \
        --workers 1 \
        --threads 4 \
        --timeout 600 \
        --max-requests 200 \
        --max-requests-jitter 30 \
        --access-logfile - \
        --error-logfile - \
        app:app'
