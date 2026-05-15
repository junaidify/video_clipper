# ── Video Auto-Clipper ──
# Multi-stage Docker build for Railway / Render / any Docker host

FROM python:3.11-slim AS base

# System deps: FFmpeg, fonts, and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    aria2 \
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

# Decode cookies from env var at startup (if set), then launch server
CMD bash -c '\
    if [ -n "$YOUTUBE_COOKIES_B64" ]; then \
        echo "$YOUTUBE_COOKIES_B64" | tr -d "\r\n " | base64 -d > /app/cookies.txt 2>/dev/null; \
        if [ -s /app/cookies.txt ]; then \
            LINES=$(wc -l < /app/cookies.txt); \
            SIZE=$(wc -c < /app/cookies.txt); \
            echo "[startup] Decoded cookies → /app/cookies.txt (${SIZE} bytes, ${LINES} lines)"; \
            head -1 /app/cookies.txt | grep -q "Netscape\|mozilla\|HTTP Cookie" \
                && echo "[startup] Cookie format looks valid" \
                || echo "[startup] WARNING: Cookie file may not be in Netscape format"; \
        else \
            echo "[startup] ERROR: Cookie decode produced empty file — check YOUTUBE_COOKIES_B64 value"; \
            rm -f /app/cookies.txt; \
        fi; \
    else \
        echo "[startup] No YOUTUBE_COOKIES_B64 set — YouTube may block downloads"; \
    fi && \
    gunicorn --bind 0.0.0.0:${PORT} \
        --workers 2 \
        --threads 4 \
        --timeout 300 \
        --max-requests 100 \
        --max-requests-jitter 20 \
        app:app'
