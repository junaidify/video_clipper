# CLIPPER — AI-Powered Video Auto-Clipper

Automatically extract the most compelling moments from long-form videos and turn them into vertical short clips optimized for TikTok, YouTube Shorts, and Instagram Reels.

Upload a video (or paste a YouTube URL), and Clipper analyzes the transcript to find hook-worthy moments — controversial takes, emotional peaks, surprising reveals — then splits the video into ready-to-post vertical clips with blurred background fill. No content is cropped or lost.

---

## Features

**Smart Clipping** — NLP-based transcript analysis scores every sentence across multiple dimensions (TF-IDF importance, hook keywords, quote detection, emotional intensity, narrative tension). Optionally falls back to LLM analysis via Groq, Gemini, or NVIDIA NIM when NLP alone finds too few hooks.

**Manual Split** — Enter start/end timestamps and get precise clips. Supports multiple timestamp pairs in one batch.

**Sequential Reels** — Split an entire video into consecutive reels of target duration. Sentence-boundary aware with 1-2 second overlap between clips for continuity. Zero frames skipped.

**Video Library** — Persistent storage for uploaded/downloaded videos. Each video gets its own folder. Reuse across multiple processing sessions without re-uploading.

**Pattern Training** — Upload examples of long-form + short-form pairs from your own content. Clipper learns your clipping style (duration preferences, position distributions, pacing patterns) and applies it to future clips.

**Clip Editor** — Dedicated post-processing page for each clip with three tools:
- **Subtitles** — Generate SRT via Whisper (local) with Gemini fallback, then burn them directly into the video with customizable font, size, color, and position.
- **Text Overlay** — Drag-and-drop text placement on any frame. Pick from 8 extracted frame thumbnails, position your text visually, then burn it into the clip.
- **Thumbnails** — Generate YouTube-style thumbnails using FFmpeg filters (template mode) or Gemini AI (picks the best frame, suggests headline and placement).

**Output Browser** — Folder-grid view of all generated clips organized by source video. Click a folder to see its clips with one-click access to the editor.

---

## Project Structure

```
video_clipper/
│
├── app.py                  ← Flask web server (routes, API, job orchestration)
├── main.py                 ← CLI entry point (headless pipeline)
├── config.py               ← Dataclass configs (pipeline, transcriber, analyzer, clipper)
│
├── transcriber.py          ← Whisper-based audio → transcript with timestamps
├── analyzer.py             ← NLP scoring engine (TF-IDF, hooks, quotes, emotion)
├── patterns.py             ← Creator pattern knowledge base (MrBeast, podcast, etc.)
├── llm_analyzer.py         ← LLM fallback via Groq / Gemini / NVIDIA NIM
│
├── clipper.py              ← FFmpeg video splitting with 9:16 vertical fit
├── manual_clipper.py       ← Timestamp-based manual splitting
├── sequential_clipper.py   ← Full-video sequential reel splitting
│
├── subtitle_generator.py   ← SRT generation + FFmpeg subtitle/text burning
├── thumbnail_generator.py  ← Frame extraction + template/AI thumbnail generation
│
├── downloader.py           ← yt-dlp wrapper for YouTube/URL downloads
├── library.py              ← Video library persistence (JSON index + folder management)
├── trainer.py              ← Pattern training from user-provided clip examples
│
├── templates/
│   ├── landing.html        ← Marketing landing page
│   ├── app.html            ← Main application dashboard (all panels)
│   ├── editor.html         ← Clip post-processing editor
│   └── index.html          ← Entry redirect
│
├── static/                 ← Static assets (CSS, JS, images)
│
├── uploads/                ← Temporary video uploads (gitignored)
├── clips_output/           ← Generated clips organized by job ID (gitignored)
├── video_library/          ← Persistent video library (gitignored)
├── training_sessions/      ← Training data from pattern learning (gitignored)
│
├── Dockerfile              ← Production Docker image (Ubuntu + FFmpeg + gunicorn)
├── railway.toml            ← Railway deployment config
├── requirements.txt        ← Python dependencies
├── .env.example            ← Environment variable template
├── .gitignore              ← Git exclusions
└── .dockerignore           ← Docker build exclusions
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Flask |
| Video Processing | FFmpeg (system binary) |
| Transcription | OpenAI Whisper (local, no API calls) |
| NLP Analysis | TF-IDF, pattern matching, hook detection |
| LLM Fallback | Groq, Google Gemini, NVIDIA NIM (optional) |
| Frontend | Vanilla HTML/CSS/JS (no build step) |
| Download | yt-dlp |
| Production Server | Gunicorn |
| Deployment | Docker, Railway |

---

## Prerequisites

- **Python 3.10+**
- **FFmpeg** — must be installed and available on PATH
  - Windows: `winget install FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/video-clipper.git
cd video-clipper

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys (optional — NLP mode works without them)

# 5. Run the app
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FLASK_SECRET_KEY` | No | Session secret (defaults to dev key) |
| `PORT` | No | Server port (default: 5000) |
| `FLASK_DEBUG` | No | Enable debug mode (default: false) |
| `GROQ_API_KEY` | No | Groq API key for LLM fallback |
| `GEMINI_API_KEY` | No | Google Gemini API key for LLM fallback + AI thumbnails + AI subtitles |
| `NVIDIA_API_KEY` | No | NVIDIA NIM API key for LLM fallback |
| `LLM_PROVIDER` | No | Preferred LLM: `groq`, `gemini`, or `nvidia` (default: groq) |

All LLM keys are optional. The NLP-based analyzer works without any API keys. LLM analysis activates only as a fallback when NLP scoring finds too few hooks.

---

## API Endpoints

### Library
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/library/list` | List all videos in library |
| POST | `/api/library/add` | Add video (URL or file upload) |
| DELETE | `/api/library/delete` | Remove video from library |

### Clipping
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/smart-clip` | Run smart analysis + auto-clip |
| POST | `/api/manual-clip` | Split by timestamps |
| POST | `/api/sequential-clip` | Sequential full-video split |
| GET | `/api/job/<job_id>` | Poll job progress |
| GET | `/api/clips/list/<job_id>` | List clips for a job |

### Post-Processing
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/clips/frame` | Extract single frame at timestamp |
| POST | `/api/clips/frames` | Extract multiple frame thumbnails |
| POST | `/api/clips/subtitle` | Generate + burn subtitles |
| POST | `/api/clips/overlay` | Burn text overlay at position |
| POST | `/api/clips/thumbnail` | Generate thumbnail (template or AI) |

### Training
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/training/start` | Start a training session |
| POST | `/api/training/add-clip` | Add example clip to session |
| POST | `/api/training/finish` | Finalize training, extract patterns |

---

## How Smart Clipping Works

1. **Download/Upload** — Video enters the library via YouTube URL or direct file upload.
2. **Transcribe** — Whisper extracts word-level timestamps from the audio track locally (no API calls).
3. **Analyze** — The NLP engine scores every transcript segment across six dimensions:
   - TF-IDF importance (rare, meaningful words)
   - Hook keyword detection (trigger words signaling high-value content)
   - Quote and emphasis detection
   - Emotional intensity scoring
   - Narrative tension arcs
   - Creator pattern matching (learned from top channels)
4. **LLM Fallback** — If NLP finds fewer hooks than the configured minimum, the transcript is sent to Groq/Gemini/NVIDIA for a second opinion.
5. **Clip** — FFmpeg splits the video at detected boundaries with smart adjustment to avoid cutting mid-sentence. Each clip gets 9:16 vertical formatting with blurred background fill.
6. **Post-Process** — Open any clip in the editor to add subtitles, text overlays, or generate thumbnails.

---

## Deployment (Railway)

The project includes production-ready deployment files for Railway:

```bash
# Push to GitHub first, then:
# 1. Go to railway.app → New Project → Deploy from GitHub
# 2. Set environment variables in the dashboard:
#    GROQ_API_KEY, GEMINI_API_KEY, FLASK_SECRET_KEY
# 3. Add a volume mounted at /app/video_library for persistence
# 4. Deploy — Railway auto-detects the Dockerfile
```

The Dockerfile installs FFmpeg, system fonts, and runs the app via gunicorn with a 300-second timeout to handle long video processing jobs.

See `Dockerfile`, `railway.toml`, and `.dockerignore` for the full configuration.

---

## CLI Usage

For headless / scripted usage without the web UI:

```bash
python main.py --input video.mp4 --output ./clips --min-duration 15 --max-duration 60
```

---

## License

MIT
