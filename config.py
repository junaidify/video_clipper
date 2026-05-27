"""
Configuration for Video Auto-Clipper
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TranscriberConfig:
    """Whisper transcription settings."""
    model_size: str = "base"  # tiny, base, small, medium, large
    language: Optional[str] = None  # None = auto-detect
    device: str = "auto"  # auto (GPU if available), cpu, or cuda


@dataclass
class AnalyzerConfig:
    """Content analysis settings."""
    # Minimum hook score (0-1) to qualify as a clip candidate
    min_hook_score: float = 0.4
    # Max number of clips to extract
    max_clips: int = 10
    # Weights for scoring components
    tfidf_weight: float = 0.30
    quote_weight: float = 0.20
    keyword_weight: float = 0.25
    sentiment_weight: float = 0.15
    position_weight: float = 0.10
    # Keywords that signal high-value content
    hook_keywords: list = field(default_factory=lambda: [
        "secret", "important", "key", "amazing", "incredible", "shocking",
        "truth", "real", "actually", "believe", "mistake", "wrong",
        "never", "always", "must", "hack", "tip", "trick", "strategy",
        "why", "how", "what if", "imagine", "listen", "here's the thing",
        "pay attention", "most people", "nobody tells you", "the problem",
        "the solution", "game changer", "breakthrough", "finally",
        "controversial", "unpopular opinion", "hot take", "fact",
        "proven", "research shows", "studies show", "data shows",
        "million", "billion", "percent", "guarantee", "promise",
        "remember", "don't forget", "critical", "essential", "vital",
    ])


@dataclass
class ClipperConfig:
    """Video clipping settings."""
    # Clip duration bounds (seconds)
    min_clip_duration: int = 15
    max_clip_duration: int = 60
    # Padding around detected hook (seconds)
    pre_hook_padding: int = 3
    post_hook_padding: int = 5
    # Output format
    output_format: str = "mp4"
    # Fit to 9:16 vertical for TikTok/Reels/Shorts (scale + blurred background, no crop)
    crop_vertical: bool = True
    # Video quality (CRF value, lower = better, 18-28 reasonable range)
    video_quality: int = 23
    # Output resolution for vertical crops (width x height)
    vertical_width: int = 1080
    vertical_height: int = 1920
    # Fade in/out duration (seconds)
    fade_duration: float = 0.5
    # Merge overlapping clips
    merge_overlapping: bool = True
    # Minimum gap between clips (seconds) — closer clips get merged
    min_gap_between_clips: int = 5


@dataclass
class PipelineConfig:
    """Master config combining all modules."""
    input_video: str = ""
    output_dir: str = "./clips_output"
    transcriber: TranscriberConfig = field(default_factory=TranscriberConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    clipper: ClipperConfig = field(default_factory=ClipperConfig)
    # Save intermediate transcript + analysis JSON
    save_analysis: bool = True
    # Verbose logging
    verbose: bool = True
