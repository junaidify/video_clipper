"""
Transcriber Module
Handles audio extraction from video and transcription using OpenAI Whisper.
Includes caching, word-level timestamps, and multi-device support.
"""
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List

from video_clipper.config import TranscriberConfig

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading on every request
_whisper_cache = {}
_whisper_lock = threading.Lock()


@dataclass
class TranscriptSegment:
    """A single segment/sentence of transcribed speech."""
    id: int
    start: float          # start time in seconds
    end: float            # end time in seconds
    text: str             # transcribed text
    words: List[dict] = field(default_factory=list)  # word-level timestamps: [{"word": str, "start": float, "end": float}]

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Transcript:
    """Full transcript of a video."""
    segments: List[TranscriptSegment]
    full_text: str
    language: str
    duration: float

    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "full_text": self.full_text,
            "language": self.language,
            "duration": self.duration,
        }

    def save(self, path: str):
        """Save transcript to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Transcript saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Transcript":
        """Load transcript from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segments = [TranscriptSegment(**s) for s in data["segments"]]
        return cls(
            segments=segments,
            full_text=data["full_text"],
            language=data["language"],
            duration=data["duration"],
        )


def _get_whisper_cache_dir() -> str:
    """Return default Whisper model download directory."""
    return os.path.expanduser(os.getenv("WHISPER_CACHE_DIR", "~/.cache/whisper"))


def _get_audio_cache_key(video_path: str) -> str:
    """Generate a cache key based on file path, size, and mtime."""
    stat = os.stat(video_path)
    key_str = f"{video_path}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(key_str.encode()).hexdigest()


def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio track from video file as 16kHz mono WAV (optimal for Whisper).

    Args:
        video_path: Path to input video file.
        output_path: Optional path for output WAV. If None, uses a temp file.

    Returns:
        Path to the extracted audio file.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    cmd = [
        "ffmpeg",
        "-y",               # overwrite output
        "-i", video_path,   # input file
        "-vn",              # disable video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",     # 16kHz sample rate (Whisper standard)
        "-ac", "1",         # mono
        output_path
    ]

    logger.info(f"Extracting audio: {video_path} -> {output_path}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    if result.returncode != 0:
        logger.error(f"FFmpeg audio extraction failed: {result.stderr}")
        raise RuntimeError(f"FFmpeg failed with exit code {result.returncode}: {result.stderr}")

    return output_path


def resolve_device(device_setting: str) -> str:
    """Resolve 'auto' to 'cuda' if available, otherwise 'cpu'."""
    if device_setting == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_setting


def _get_whisper_model(model_size: str, device: str):
    """Get or load a cached Whisper model (thread-safe)."""
    import whisper

    resolved_device = resolve_device(device)
    cache_key = f"{model_size}_{resolved_device}"

    with _whisper_lock:
        if cache_key not in _whisper_cache:
            logger.info(f"Loading Whisper model '{model_size}' on {resolved_device}...")
            _whisper_cache[cache_key] = whisper.load_model(model_size, device=resolved_device)
            logger.info(f"Whisper model '{model_size}' loaded successfully.")
        return _whisper_cache[cache_key]


def transcribe(
    video_path: str,
    config: Optional[TranscriberConfig] = None,
    cache_dir: Optional[str] = None,
) -> Transcript:
    """
    Transcribe a video file using OpenAI Whisper.

    Args:
        video_path: Path to the input video.
        config: TranscriberConfig instance (defaults if None).
        cache_dir: Optional directory to cache transcripts as JSON.

    Returns:
        Transcript object containing segments and full text.
    """
    if config is None:
        config = TranscriberConfig()

    # Check cache first
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = _get_audio_cache_key(video_path)
        cache_file = os.path.join(cache_dir, f"{cache_key}_{config.model_size}.json")
        if os.path.isfile(cache_file):
            logger.info(f"Loading cached transcript from {cache_file}")
            return Transcript.load(cache_file)

    # Extract audio to temp file
    audio_path = extract_audio(video_path)

    try:
        model = _get_whisper_model(config.model_size, config.device)

        # Transcribe options
        options = {
            "word_timestamps": True,
            "verbose": False,
        }
        if config.language:
            options["language"] = config.language
        if config.initial_prompt:
            options["initial_prompt"] = config.initial_prompt

        logger.info(f"Starting Whisper transcription (model={config.model_size})...")
        result = model.transcribe(audio_path, **options)

        # Parse segments
        segments = []
        for i, seg in enumerate(result.get("segments", [])):
            words = []
            for w in seg.get("words", []):
                words.append({
                    "word": w.get("word", "").strip(),
                    "start": round(w.get("start", 0.0), 3),
                    "end": round(w.get("end", 0.0), 3),
                    "probability": round(w.get("probability", 0.0), 3),
                })

            segments.append(TranscriptSegment(
                id=i,
                start=round(seg.get("start", 0.0), 3),
                end=round(seg.get("end", 0.0), 3),
                text=seg.get("text", "").strip(),
                words=words,
            ))

        # Total duration from last segment
        duration = segments[-1].end if segments else 0.0

        transcript = Transcript(
            segments=segments,
            full_text=result.get("text", "").strip(),
            language=result.get("language", "unknown"),
            duration=duration,
        )

        logger.info(
            f"Transcription complete: {len(segments)} segments, "
            f"{duration:.1f}s duration, language='{transcript.language}'"
        )

        # Save to cache if requested
        if cache_dir:
            transcript.save(cache_file)

        return transcript

    finally:
        # Clean up temp audio file
        if os.path.isfile(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass
