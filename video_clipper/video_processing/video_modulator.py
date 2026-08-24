"""
Video Modulator Module
Applies pixel-level and temporal transformations to break perceptual hashing:
- 3-5% scale up + crop to break spatial pixel grid
- Horizontal mirror flip
- Subtle film-grade color grading
- Dynamic film grain overlay
- Center hardcoded subtitle burn-in
- Playback speed shift (0.98x - 1.02x)

All operations use FFmpeg filters natively.
"""
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class ModulationConfig:
    """Configuration for video modulation transforms."""
    zoom_percent: float = 4.0          # 3-5% zoom to break pixel grid
    mirror_flip: bool = False          # Horizontal flip
    color_grade: str = "warm"          # "warm", "cool", "cinematic", "vibrant", "none"
    grain_overlay: bool = True         # film grain
    grain_intensity: float = 0.05      # Grain opacity (0.0-0.15)
    saturation_boost: float = 1.10     # Saturation multiplier
    black_crush: float = 0.05          # Subtle black level crush
    speed_shift: float = 1.0           # Playback speed (1.0 = unchanged, 1.02 = 2% faster)
    burn_subtitles: bool = False
    subtitle_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "zoom_percent": self.zoom_percent,
            "mirror_flip": self.mirror_flip,
            "color_grade": self.color_grade,
            "grain_overlay": self.grain_overlay,
            "grain_intensity": self.grain_intensity,
            "saturation_boost": self.saturation_boost,
            "black_crush": self.black_crush,
            "speed_shift": self.speed_shift,
            "burn_subtitles": self.burn_subtitles,
        }


@dataclass
class ModulationResult:
    """Result of video modulation."""
    output_path: str
    transforms_applied: List[str]
    original_size: int
    modulated_size: int
    duration: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "transforms_applied": self.transforms_applied,
            "original_size": self.original_size,
            "modulated_size": self.modulated_size,
            "duration": self.duration,
            "success": self.success,
            "error": self.error,
        }


COLOR_GRADES = {
    "warm": {
        "brightness": 0.02,
        "contrast": 1.05,
        "saturation": 1.12,
        "gamma_r": 1.05,
        "gamma_b": 0.95,
    },
    "cool": {
        "brightness": 0.01,
        "contrast": 1.08,
        "saturation": 1.05,
        "gamma_r": 0.95,
        "gamma_b": 1.08,
    },
    "cinematic": {
        "brightness": -0.02,
        "contrast": 1.15,
        "saturation": 0.90,
        "gamma_r": 1.02,
        "gamma_b": 1.02,
    },
    "vibrant": {
        "brightness": 0.03,
        "contrast": 1.10,
        "saturation": 1.25,
        "gamma_r": 1.0,
        "gamma_b": 1.0,
    },
    "none": None,
}


def modulate_video(
    video_path: str,
    output_path: str,
    config: Optional[ModulationConfig] = None,
) -> ModulationResult:
    """
    Apply pixel-level modulation transforms to a video clip.

    Args:
        video_path: Source video file.
        output_path: Destination modulated video file.
        config: ModulationConfig instance.

    Returns:
        ModulationResult
    """
    if config is None:
        config = ModulationConfig()

    if not os.path.isfile(video_path):
        return ModulationResult(output_path, [], 0, 0, 0, False, "Video file not found")

    original_size = os.path.getsize(video_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    transforms_applied = []
    video_filters = []
    audio_filters = []

    # 1. Zoom
    if config.zoom_percent > 0:
        scale_factor = 1.0 + (config.zoom_percent / 100.0)
        video_filters.append(
            f"scale=iw*{scale_factor}:ih*{scale_factor},"
            f"crop=iw/{scale_factor}:ih/{scale_factor}"
        )
        transforms_applied.append(f"zoom_{config.zoom_percent}%")

    # 2. Mirror flip
    if config.mirror_flip:
        video_filters.append("hflip")
        transforms_applied.append("mirror_flip")

    # 3. Color grading
    grade = COLOR_GRADES.get(config.color_grade)
    if grade:
        eq_parts = [
            f"brightness={grade['brightness']}",
            f"contrast={grade['contrast']}",
            f"saturation={grade['saturation'] * config.saturation_boost}",
            f"gamma_r={grade['gamma_r']}",
            f"gamma_b={grade['gamma_b']}",
        ]
        video_filters.append(f"eq={':'.join(eq_parts)}")
        transforms_applied.append(f"color_{config.color_grade}")

        if config.black_crush > 0:
            crush = config.black_crush
            video_filters.append(f"curves=m='0/0 {crush}/{crush*0.6} 0.5/0.5 1/1'")
            transforms_applied.append(f"black_crush_{crush}")
    elif config.saturation_boost != 1.0:
        video_filters.append(f"eq=saturation={config.saturation_boost}")
        transforms_applied.append(f"saturation_{config.saturation_boost}")

    # 4. Film grain
    if config.grain_overlay and config.grain_intensity > 0:
        intensity = int(config.grain_intensity * 255)
        video_filters.append(f"noise=alls={intensity}:allf=t")
        transforms_applied.append(f"grain_{config.grain_intensity}")

    # 5. Subtitle burn-in
    if config.burn_subtitles and config.subtitle_path and os.path.isfile(config.subtitle_path):
        sub_path = config.subtitle_path.replace("\\", "/").replace(":", "\\:")
        video_filters.append(
            f"subtitles='{sub_path}':force_style="
            f"'Alignment=2,FontSize=22,PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,Outline=2,Shadow=1,"
            f"FontName=Arial,Bold=1,MarginV=40'"
        )
        transforms_applied.append("hardcoded_subtitles")

    # 6. Speed shift
    if config.speed_shift != 1.0 and 0.5 <= config.speed_shift <= 2.0:
        speed = config.speed_shift
        video_filters.append(f"setpts={1.0/speed}*PTS")
        audio_filters.append(f"atempo={speed}")
        transforms_applied.append(f"speed_{speed}x")

    if not transforms_applied:
        transforms_applied.append("passthrough")

    try:
        cmd = ["ffmpeg", "-y", "-i", video_path]

        if video_filters:
            cmd.extend(["-vf", ",".join(video_filters)])

        if audio_filters:
            cmd.extend(["-af", ",".join(audio_filters)])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            output_path
        ])

        logger.info(f"Modulating video: {' -> '.join(transforms_applied)}")
        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8',
            errors='replace', timeout=900
        )

        if result.returncode != 0:
            logger.error(f"Modulation failed: {result.stderr[-400:]}")
            return ModulationResult(
                output_path, transforms_applied,
                original_size, 0, 0, False,
                f"FFmpeg failed: {result.stderr[-200:]}"
            )

        modulated_size = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
        duration = _get_duration(output_path)

        logger.info(
            f"Modulation complete: {len(transforms_applied)} transforms, "
            f"{original_size/1024/1024:.1f}MB -> {modulated_size/1024/1024:.1f}MB"
        )

        return ModulationResult(
            output_path, transforms_applied,
            original_size, modulated_size,
            duration, True
        )

    except subprocess.TimeoutExpired:
        return ModulationResult(
            output_path, transforms_applied,
            original_size, 0, 0, False,
            "Modulation timed out (15 min limit)"
        )
    except Exception as e:
        logger.error(f"Modulation error: {e}")
        return ModulationResult(
            output_path, transforms_applied,
            original_size, 0, 0, False, str(e)
        )


def get_presets() -> dict:
    """Return available modulation presets for UI."""
    return {
        "stealth": {
            "label": "Stealth Mode",
            "description": "Minimal visible changes, maximum hash disruption",
            "config": ModulationConfig(
                zoom_percent=3.0,
                mirror_flip=False,
                color_grade="none",
                grain_overlay=True,
                grain_intensity=0.03,
                saturation_boost=1.02,
                black_crush=0.02,
            ).to_dict(),
        },
        "cinematic": {
            "label": "Cinematic Grade",
            "description": "Film-like color grading with subtle grain",
            "config": ModulationConfig(
                zoom_percent=4.0,
                mirror_flip=False,
                color_grade="cinematic",
                grain_overlay=True,
                grain_intensity=0.06,
                saturation_boost=1.0,
                black_crush=0.05,
            ).to_dict(),
        },
        "vibrant": {
            "label": "Vibrant Pop",
            "description": "Boosted colors, high energy feel",
            "config": ModulationConfig(
                zoom_percent=4.0,
                mirror_flip=False,
                color_grade="vibrant",
                grain_overlay=True,
                grain_intensity=0.04,
                saturation_boost=1.15,
                black_crush=0.03,
            ).to_dict(),
        },
        "warm_flip": {
            "label": "Warm + Mirror",
            "description": "Warm tones with horizontal flip",
            "config": ModulationConfig(
                zoom_percent=5.0,
                mirror_flip=True,
                color_grade="warm",
                grain_overlay=True,
                grain_intensity=0.05,
                saturation_boost=1.10,
                black_crush=0.04,
            ).to_dict(),
        },
        "maximum": {
            "label": "Maximum Disruption",
            "description": "Every transform enabled",
            "config": ModulationConfig(
                zoom_percent=5.0,
                mirror_flip=True,
                color_grade="vibrant",
                grain_overlay=True,
                grain_intensity=0.08,
                saturation_boost=1.20,
                black_crush=0.06,
                speed_shift=1.02,
            ).to_dict(),
        },
    }


def _get_duration(path: str) -> float:
    if not os.path.isfile(path):
        return 0.0
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", path],
            capture_output=True, encoding='utf-8', errors='replace'
        )
        info = json.loads(r.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0
