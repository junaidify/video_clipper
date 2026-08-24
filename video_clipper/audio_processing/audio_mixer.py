"""
Audio Mixer Module
Mixes generated voiceover/commentary audio with source video audio.
Supports audio replacement (clean mute) and ducking (background audio dips under speech).
"""
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AudioMixResult:
    """Result of audio mixing."""
    output_path: str
    duration: float
    mix_mode: str          # 'replace' or 'duck'
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "duration": self.duration,
            "mix_mode": self.mix_mode,
            "success": self.success,
            "error": self.error,
        }


def mix_commentary(
    video_path: str,
    commentary_audio_path: str,
    output_path: str,
    mix_mode: str = "duck",
    original_volume: float = 0.15,
    commentary_volume: float = 1.0,
    duck_threshold: float = 0.03,
    duck_ratio: float = 4.0,
) -> AudioMixResult:
    """
    Mix commentary audio track with the video.

    Args:
        video_path: Path to source video.
        commentary_audio_path: Path to voiceover audio file.
        output_path: Destination path for mixed video.
        mix_mode: 'duck' (mix with ducked source audio) or 'replace' (source audio muted).
        original_volume: Base volume of original audio when commentary is quiet (0.0-1.0).
        commentary_volume: Volume multiplier for commentary (0.0-1.5).
        duck_threshold: Voice detection threshold for ducking (0.01-0.1).
        duck_ratio: How much to compress original audio during speech (2-8).

    Returns:
        AudioMixResult
    """
    if not os.path.isfile(video_path):
        return AudioMixResult(output_path, 0, mix_mode, False, "Source video not found")
    if not os.path.isfile(commentary_audio_path):
        return AudioMixResult(output_path, 0, mix_mode, False, "Commentary audio not found")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if mix_mode == "replace":
        return _replace_audio(video_path, commentary_audio_path, output_path, commentary_volume)
    else:
        return _mix_audio(
            video_path, commentary_audio_path, output_path,
            original_volume, commentary_volume,
            duck_threshold, duck_ratio
        )


def _replace_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    audio_volume: float = 1.0,
) -> AudioMixResult:
    """Replace original audio with commentary voiceover."""
    logger.info(f"Replacing audio in {video_path} with {audio_path}")

    try:
        vol_filter = f"volume={audio_volume}" if audio_volume != 1.0 else "anull"

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-af", vol_filter,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ]

        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8',
            errors='replace', timeout=180
        )

        if result.returncode != 0:
            logger.error(f"Audio replace failed: {result.stderr[-300:]}")
            return AudioMixResult(output_path, 0, "replace", False, result.stderr[-200:])

        duration = _get_duration(output_path)
        return AudioMixResult(output_path, duration, "replace", True)

    except Exception as e:
        logger.error(f"Audio replace error: {e}")
        return AudioMixResult(output_path, 0, "replace", False, str(e))


def _mix_audio(
    video_path: str,
    commentary_path: str,
    output_path: str,
    original_volume: float,
    commentary_volume: float,
    duck_threshold: float,
    duck_ratio: float,
) -> AudioMixResult:
    """Mix original audio and commentary with sidechain auto-ducking."""
    logger.info(
        f"Mixing audio with ducking: orig_vol={original_volume}, "
        f"commentary_vol={commentary_volume}, threshold={duck_threshold}"
    )

    try:
        has_orig_audio = _has_audio_stream(video_path)

        if not has_orig_audio:
            return _replace_audio(video_path, commentary_path, output_path, commentary_volume)

        filter_complex = (
            f"[0:a]volume={original_volume},aresample=48000[orig];"
            f"[1:a]volume={commentary_volume},aresample=48000[comm];"
            f"[orig][comm]sidechaincompress="
            f"threshold={duck_threshold}:ratio={duck_ratio}:"
            f"attack=0.01:release=0.3:level_in=1:level_sc=1[ducked];"
            f"[comm][ducked]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,"
            f"alimiter=limit=0.95[aout]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", commentary_path,
            "-filter_complex", filter_complex,
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ]

        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8',
            errors='replace', timeout=180
        )

        if result.returncode != 0:
            logger.warning(f"Sidechain ducking failed, falling back to simple mix: {result.stderr[-200:]}")
            return _simple_mix_fallback(video_path, commentary_path, output_path, original_volume, commentary_volume)

        duration = _get_duration(output_path)
        return AudioMixResult(output_path, duration, "duck", True)

    except Exception as e:
        logger.error(f"Audio mix error: {e}")
        return AudioMixResult(output_path, 0, "duck", False, str(e))


def _simple_mix_fallback(
    video_path: str,
    commentary_path: str,
    output_path: str,
    original_volume: float,
    commentary_volume: float,
) -> AudioMixResult:
    """Fallback: simple volume-weighted mix without sidechain compression."""
    try:
        filter_complex = (
            f"[0:a]volume={original_volume * 0.5}[a0];"
            f"[1:a]volume={commentary_volume}[a1];"
            f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=2,"
            f"alimiter=limit=0.95[aout]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", commentary_path,
            "-filter_complex", filter_complex,
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ]

        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8',
            errors='replace', timeout=180
        )

        if result.returncode == 0:
            duration = _get_duration(output_path)
            return AudioMixResult(output_path, duration, "duck_fallback", True)
        else:
            return AudioMixResult(output_path, 0, "duck_fallback", False, result.stderr[-200:])

    except Exception as e:
        return AudioMixResult(output_path, 0, "duck_fallback", False, str(e))


def _has_audio_stream(path: str) -> bool:
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "json", path
        ]
        r = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        data = json.loads(r.stdout)
        return len(data.get("streams", [])) > 0
    except Exception:
        return False


def _get_duration(path: str) -> float:
    if not os.path.isfile(path):
        return 0.0
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json", path
        ]
        r = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        info = json.loads(r.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0
