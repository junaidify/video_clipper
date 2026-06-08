"""
Transcriber Module
Extracts audio from video and transcribes using OpenAI Whisper
with word/segment-level timestamps.
"""
import json
import logging
import os
import subprocess
import tempfile
import threading
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Suppress harmless Whisper Triton/CUDA fallback warnings
# These fire on every transcription when no CUDA GPU is available
# and just say "falling back to slower CPU implementation" — not errors
warnings.filterwarnings("ignore", message=".*Triton.*falling back.*")
warnings.filterwarnings("ignore", message=".*Failed to launch Triton kernels.*")

logger = logging.getLogger(__name__)


def resolve_device(device: str = "auto") -> str:
    """
    Resolve 'auto' to the best available device.
    Returns 'cuda' if an NVIDIA GPU with CUDA is available, else 'cpu'.
    """
    if device and device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name} — using CUDA")
            return "cuda"
    except ImportError:
        pass
    logger.info("No CUDA GPU detected — using CPU")
    return "cpu"


# ─── Whisper Model Cache ───
# Loading the model takes 5-15 seconds. Cache it globally so subsequent
# transcriptions reuse the already-loaded model instantly.
_whisper_cache = {"model": None, "size": None, "device": None}
_whisper_lock = threading.Lock()


@dataclass
class TranscriptSegment:
    """A single segment of transcribed text with timestamps."""
    id: int
    start: float  # seconds
    end: float    # seconds
    text: str
    words: list   # word-level timestamps if available

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Transcript:
    """Full transcript with all segments."""
    segments: list  # List[TranscriptSegment]
    full_text: str
    language: str
    duration: float  # total video duration in seconds

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "duration": self.duration,
            "full_text": self.full_text,
            "segments": [s.to_dict() for s in self.segments],
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Transcript saved to {path}")


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path
    ]
    try:
        # Use encoding='utf-8' + errors='replace' to handle non-ASCII filenames on Windows
        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8', errors='replace', check=True
        )
        if not result.stdout:
            logger.warning("ffprobe returned empty output")
            return 0.0
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError, TypeError) as e:
        logger.warning(f"ffprobe failed, duration unknown: {e}")
        return 0.0


def extract_audio(video_path: str, audio_path: str) -> str:
    """Extract audio from video as WAV for Whisper processing."""
    logger.info(f"Extracting audio from: {video_path}")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # WAV format
        "-ar", "16000",           # 16kHz sample rate (Whisper optimal)
        "-ac", "1",               # mono
        audio_path
    ]
    # Use encoding='utf-8' + errors='replace' to handle non-ASCII paths on Windows
    subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', check=True)
    logger.info(f"Audio extracted to: {audio_path}")
    return audio_path


def _get_cache_path(video_path: str, model_size: str) -> str:
    """Build the cache file path for a video's transcript."""
    video = Path(video_path)
    return str(video.parent / f"{video.stem}_transcript_{model_size}.json")


def _load_cached_transcript(cache_path: str) -> Optional[Transcript]:
    """Load a cached transcript from disk if it exists and is valid."""
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate required fields
        if not data.get("segments"):
            return None

        segments = []
        for s in data["segments"]:
            segments.append(TranscriptSegment(
                id=s["id"],
                start=s["start"],
                end=s["end"],
                text=s["text"],
                words=s.get("words", []),
            ))

        transcript = Transcript(
            segments=segments,
            full_text=data.get("full_text", ""),
            language=data.get("language", "unknown"),
            duration=data.get("duration", 0),
        )
        logger.info(f"Loaded cached transcript: {cache_path} ({len(segments)} segments)")
        return transcript

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Invalid transcript cache, will re-transcribe: {e}")
        return None


def _save_transcript_cache(transcript: Transcript, cache_path: str):
    """Save transcript to disk for future reuse."""
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(transcript.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Transcript cached: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to cache transcript: {e}")


def transcribe(video_path: str, model_size: str = "base",
               language: Optional[str] = None,
               device: str = "auto") -> Transcript:
    """
    Transcribe video using OpenAI Whisper.
    Caches results alongside the video file for instant reuse.

    Args:
        video_path: Path to the input video file
        model_size: Whisper model size (tiny/base/small/medium/large)
        language: Language code or None for auto-detect
        device: 'auto' (GPU if available), 'cpu', or 'cuda'

    Returns:
        Transcript object with segments and timestamps
    """
    device = resolve_device(device)

    video_path = str(Path(video_path).resolve())

    # ── Check cache first ──
    cache_path = _get_cache_path(video_path, model_size)
    cached = _load_cached_transcript(cache_path)
    if cached:
        return cached

    try:
        import whisper
    except ImportError:
        raise ImportError(
            "OpenAI Whisper is required. Install it with:\n"
            "  pip install openai-whisper\n"
            "Also requires ffmpeg installed on your system."
        )

    duration = get_video_duration(video_path)

    # Extract audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        extract_audio(video_path, audio_path)

        # Load Whisper model (cached — first load takes 5-15s, subsequent calls are instant)
        with _whisper_lock:
            if (_whisper_cache["model"] is not None
                    and _whisper_cache["size"] == model_size
                    and _whisper_cache["device"] == device):
                logger.info(f"Reusing cached Whisper model: {model_size}")
                model = _whisper_cache["model"]
            else:
                logger.info(f"Loading Whisper model: {model_size} (first load, will be cached)")
                model = whisper.load_model(model_size, device=device)
                _whisper_cache["model"] = model
                _whisper_cache["size"] = model_size
                _whisper_cache["device"] = device

        # Transcribe with word-level timestamps
        logger.info("Transcribing audio (this may take a while)...")
        options = {"word_timestamps": True, "verbose": False}
        if language:
            options["language"] = language

        result = model.transcribe(audio_path, **options)

        # Build structured transcript
        segments = []
        for i, seg in enumerate(result["segments"]):
            words = []
            if "words" in seg:
                words = [
                    {"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
                    for w in seg["words"]
                ]

            segments.append(TranscriptSegment(
                id=i,
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                words=words,
            ))

        detected_lang = result.get("language", "unknown")
        full_text = " ".join(s.text for s in segments)

        logger.info(
            f"Transcription complete: {len(segments)} segments, "
            f"language={detected_lang}, duration={duration:.1f}s"
        )

        transcript = Transcript(
            segments=segments,
            full_text=full_text,
            language=detected_lang,
            duration=duration or (segments[-1].end if segments else 0),
        )

        # ── Save to cache for future reuse ──
        _save_transcript_cache(transcript, cache_path)

        return transcript

    finally:
        # Cleanup temp audio
        Path(audio_path).unlink(missing_ok=True)
