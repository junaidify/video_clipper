# Video Auto-Clipper — Project Tracker

## Overview
Analyzes video content via transcript, detects hook-worthy moments (key quotes, interesting points, emotional peaks), and **automatically splits the video into short vertical clips** for TikTok / Reels / YouTube Shorts.

## Architecture

```
video_clipper/
├── app.py                    # Flask web server (main entry for web UI)
├── main.py                   # CLI entry point (standalone usage)
├── config.py                 # Dataclass configs for all modules
├── core/
│   ├── transcriber.py        # Whisper-based audio transcription
│   ├── analyzer.py           # NLP content analysis (TF-IDF, keywords, sentiment)
│   ├── clipper.py            # FFmpeg video splitting + 9:16 scale-to-fit (blurred bg)
│   ├── patterns.py           # Creator patterns knowledge base (viral structures)
│   └── llm_analyzer.py       # Optional LLM fallback (Groq, Gemini, NVIDIA)
├── modes/
│   ├── manual_clipper.py     # Timestamp-based splitting
│   ├── sequential_clipper.py # Consecutive reels generation
│   └── full_video.py         # Full video processing mode
├── post/
│   ├── subtitle_generator.py # Subtitle generation
│   ├── thumbnail_generator.py# Thumbnail generation
│   ├── video_modulator.py    # Hash-breaking transforms
│   └── video_editor.py       # Scene assembly
├── media/
│   ├── downloader.py         # YouTube/URL download via yt-dlp
│   ├── library.py            # Video library management
│   ├── audio_mixer.py        # Audio mixing
│   ├── music_mixer.py        # Background music
│   └── tts_engine.py         # Text-to-speech
├── ai/
│   ├── commentary.py         # AI narration scripts
│   ├── script_generator.py   # Script generation
│   ├── trend_scout.py        # Trending topics
│   └── visual_engine.py      # Stock footage
├── publish/
│   ├── youtube_uploader.py   # YouTube upload
│   └── content_factory.py    # Orchestrator
├── training/
│   └── trainer.py            # Pattern training
├── tests/                    # Test suite
├── templates/
│   └── index.html            # Web UI frontend
├── .env                      # API keys — NOT committed
├── .env.example              # Template for .env
├── requirements.txt          # Python dependencies
├── uploads/                  # Temp storage for uploaded/downloaded videos
└── clips_output/             # Generated clips organized by job ID
```

## Pipeline (3 stages)
1. **Transcribe** — Whisper extracts word-level timestamps from audio
2. **Analyze** — NLP scores segments across 5 dimensions (TF-IDF, quotes, keywords, sentiment, position). LLM fallback kicks in only if NLP finds <2 candidates.
3. **Split** — FFmpeg cuts video into separate MP4 files with smart boundaries, 9:16 crop, fade transitions

## API Keys (.env)
| Key | Provider | Required? | Purpose |
|-----|----------|-----------|---------|
| `GROQ_API_KEY` | Groq | Optional | LLM fallback via Llama/Mixtral |
| `GEMINI_API_KEY` | Google | Optional | LLM fallback via Gemini |
| `NVIDIA_API_KEY` | NVIDIA NIM | Optional | LLM fallback via NVIDIA models |

NLP-only mode works without any API keys. LLM is fallback only.

## How to Run

### Web UI (recommended)
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

### CLI
```bash
python main.py video.mp4 --output ./clips --max-clips 5
```

## System Requirements
- Python 3.9+
- FFmpeg installed and on PATH
- ~2GB disk for Whisper model (first run downloads it)

## Changelog

### v2.3 — 2026-05-06
- Fixed: Unicode crash on Windows when video titles contain non-ASCII characters (Hindi, Arabic, Chinese, etc.)
- Downloader now uses video ID as filename instead of title (avoids encoding issues entirely)
- All subprocess calls (ffprobe, ffmpeg) now use `encoding='utf-8', errors='replace'` instead of default cp1252
- Fixed: `json.loads(None)` crash when ffprobe stdout was empty due to encoding failure

### v2.2 — 2026-05-06
- **No-crop video fitting**: Replaced center-crop with scale-to-fit + blurred background fill (YouTube Shorts style). No content is lost — video shrinks to fit 9:16, empty space filled with blurred version of the video itself.
- **Creator patterns intelligence** (`patterns.py`): Baked in viral clipping patterns from top YouTube channels (MrBeast, Hormozi, Huberman, podcast clips, TED, etc.). 6 pattern categories: hook openers, value bombs, emotional peaks, structural signals, clip arc templates, transition markers. Integrated into analyzer scoring at 40% weight alongside NLP.
- **Clip arc detection**: Identifies which viral template a clip matches (hook→value→CTA, story→climax→lesson, claim→proof→takeaway, etc.) and gives bonus score.

### v2.1 — 2026-05-06
- Fixed: YouTube download crash when FFmpeg is not installed
- Downloader now auto-detects FFmpeg and falls back to single combined stream (lower quality but works)
- Web UI shows FFmpeg warning banner when missing
- Settings API returns `ffmpeg_installed` status
- FFmpeg badge added to settings panel

### v2.0 — 2026-05-05
- Added Flask web UI (upload video or paste YouTube URL)
- Added YouTube/URL download support via yt-dlp
- Added LLM fallback analyzer (Groq, Gemini, NVIDIA)
- Added .env for API key management
- Added real-time progress tracking in web UI
- Added video preview + download for generated clips

### v1.0 — 2026-05-05
- Initial NLP-based content analyzer (TF-IDF, keywords, sentiment, quotes, position scoring)
- Whisper transcription with word-level timestamps
- FFmpeg video splitting with smart boundary detection
- 9:16 vertical crop for TikTok/Reels/Shorts
- CLI interface with configurable parameters

## Known Limitations
- Whisper transcription can be slow on CPU for long videos (use `--model tiny` or `--device cuda`)
- NLP scoring works best on English content
- No face-tracking for crop centering (uses center-crop)
- Single-threaded clip extraction (clips are cut sequentially)

## Future Ideas
- Face detection for smart crop positioning
- Subtitle burn-in on clips
- Batch processing (multiple videos)
- Auto-upload to TikTok/YouTube via API
- Thumbnail generation for each clip
