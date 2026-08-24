"""
Configuration Module for Video Auto-Clipper.
Centralizes dataclasses and default parameters across the entire processing pipeline.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TranscriberConfig:
    """Settings for audio extraction and Whisper transcription."""
    model_size: str = "base"          # tiny, base, small, medium, large
    language: Optional[str] = None     # auto-detect if None, e.g. "en", "hi", "hinglish"
    device: str = "auto"              # "auto", "cuda", "cpu"
    initial_prompt: str = ""          # optional Whisper prompt context


@dataclass
class AnalyzerConfig:
    """Settings for transcript analysis and hook detection."""
    min_hook_score: float = 0.4       # minimum composite score (0-1) to qualify as a clip
    max_clips: int = 10               # max number of clips to extract from one video
    window_size: int = 30             # sliding window size in seconds
    window_stride: int = 5            # step size for sliding window in seconds

    # Scoring weights (sum should equal 1.0)
    tfidf_weight: float = 0.25
    quote_weight: float = 0.20
    keyword_weight: float = 0.25
    sentiment_weight: float = 0.15
    position_weight: float = 0.15

    # Hook keywords to look for
    hook_keywords: List[str] = field(default_factory=lambda: [
        "secret", "secret to", "truth about", "never", "always",
        "mistake", "everyone thinks", "nobody talks about", "here's why",
        "the real reason", "stop doing", "start doing", "how to",
        "number one", "first thing", "biggest", "most important",
        "changed my life", "game changer", "warning", "don't do this",
        "hack", "trick", "blueprint", "framework", "formula",
        "lesson", "rule", "step", "actually", "literally",
        "insane", "crazy", "unbelievable", "amazing", "shocking"
    ])


@dataclass
class ClipperConfig:
    """Settings for FFmpeg cutting, scaling, and video export."""
    min_clip_duration: int = 15       # minimum clip length in seconds
    max_clip_duration: int = 60       # maximum clip length in seconds (for YouTube Shorts / TikTok)
    pre_hook_padding: int = 3         # seconds before the hook to include for context
    post_hook_padding: int = 5        # seconds after the hook before ending
    output_format: str = "mp4"        # container format

    # FFmpeg encoding
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    video_bitrate: str = "4M"
    audio_bitrate: str = "192k"
    video_quality: int = 23           # CRF (18 = visually lossless, 23 = default, 28 = small)
    video_preset: str = "veryfast"    # ultrafast, superfast, veryfast, faster, fast, medium

    # 9:16 vertical crop settings
    crop_vertical: bool = True        # fit to 9:16 for Shorts/Reels/TikTok
    vertical_width: int = 1080
    vertical_height: int = 1920

    # Transitions
    fade_duration: float = 0.5        # seconds for fade-in / fade-out (0 = disabled)


@dataclass
class PipelineConfig:
    """Top-level configuration for the entire video clipping pipeline."""
    input_video: str = ""
    output_dir: str = "./clips_output"
    save_analysis: bool = True        # whether to save JSON analysis alongside clips
    verbose: bool = True

    transcriber: TranscriberConfig = field(default_factory=TranscriberConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    clipper: ClipperConfig = field(default_factory=ClipperConfig)
