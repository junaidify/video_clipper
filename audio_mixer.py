"""
Audio Mixer Module
Combines commentary audio with video — either full replacement or ducked mix.
Uses FFmpeg for all audio operations.
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
class MixResult:
    """Result of audio mixing."""
    output_path: str
    mode: str               # 'replace' or 'mix'
    original_volume: float  # 0.0 for replace, 0.0-1.0 for mix
    duration: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "mode": self.mode,
            "original_volume": self.original_volume,
            "duration": self.duration,
            "success": self.success,
            "error": self.error,
        }


def mix_commentary(video_path: str,
                   commentary_audio_path: str,
                   output_path: str,
                   mode: str = "replace",
                   original_volume: float = 0.2,
                   commentary_volume: float = 1.0) -> MixResult:
    """
    Mix commentary audio with video.

    Args:
        video_path: Source video file
        commentary_audio_path: Generated commentary audio
        output_path: Where to save the output
        mode: 'replace' (mute original) or 'mix' (duck original)
        original_volume: Volume of original audio in mix mode (0.0-1.0)
        commentary_volume: Volume of commentary (0.0-1.0)

    Returns:
        MixResult
    """
    if not os.path.isfile(video_path):
        return MixResult(output_path, mode, original_volume, 0, False, "Video file not found")
    if not os.path.isfile(commentary_audio_path):
        return MixResult(output_path, mode, original_volume, 0, False, "Commentary audio not found")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        if mode == "replace":
            result = _replace_audio(video_path, commentary_audio_path, output_path,
                                     commentary_volume)
        elif mode == "mix":
            result = _mix_audio(video_path, commentary_audio_path, output_path,
                                original_volume, commentary_volume)
        else:
            return MixResult(output_path, mode, original_volume, 0, False,
                             f"Unknown mode: {mode}")

        if result:
            duration = _get_duration(output_path)
            logger.info(f"Audio mixed ({mode}): {output_path} ({duration:.1f}s)")
            return MixResult(output_path, mode, original_volume, duration, True)
        else:
            return MixResult(output_path, mode, original_volume, 0, False,
                             "FFmpeg mixing failed")

    except Exception as e:
        logger.error(f"Audio mix error: {e}")
        return MixResult(output_path, mode, original_volume, 0, False, str(e))


def _replace_audio(video_path: str, audio_path: str, output_path: str,
                    commentary_volume: float) -> bool:
    """
    Nuclear audio replacement — ZERO original audio bleed guaranteed.

    Two-step process:
      1. Extract video-only to a temp file (no audio track at all)
      2. Mux the silent video with commentary audio

    This makes bleed-through physically impossible because the
    intermediate file has NO audio stream to leak.
    """
    import tempfile

    tmp_dir = tempfile.mkdtemp()
    silent_video = os.path.join(tmp_dir, "video_only.mp4")

    try:
        # ── Step 1: Strip ALL audio from the video ──
        strip_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-an",              # absolutely no audio
            "-c:v", "copy",     # don't re-encode video
            silent_video
        ]
        strip_result = subprocess.run(
            strip_cmd, capture_output=True, encoding='utf-8',
            errors='replace', timeout=120
        )
        if strip_result.returncode != 0:
            logger.error(f"Audio strip failed: {strip_result.stderr[-300:]}")
            return False

        # Verify: the temp file must have ZERO audio streams
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "a", silent_video
        ]
        probe_result = subprocess.run(
            probe_cmd, capture_output=True, encoding='utf-8', errors='replace'
        )
        try:
            probe_data = json.loads(probe_result.stdout)
            if probe_data.get("streams"):
                logger.error("Audio strip failed: temp file still has audio streams!")
                return False
        except (json.JSONDecodeError, TypeError):
            pass  # probe failed, proceed anyway — the -an flag is reliable

        # ── Step 2: Mux silent video + commentary ──
        vol_val = commentary_volume * 1.5
        vol_filter = f"volume={vol_val},alimiter=limit=0.95"

        mux_cmd = [
            "ffmpeg", "-y",
            "-i", silent_video,     # video with NO audio
            "-i", audio_path,       # commentary only
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-af", vol_filter,
            "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ]
        mux_result = subprocess.run(
            mux_cmd, capture_output=True, encoding='utf-8',
            errors='replace', timeout=180
        )
        if mux_result.returncode != 0:
            logger.error(f"Mux failed: {mux_result.stderr[-300:]}")
            return False

        logger.info("Nuclear replace complete — zero original audio in output")
        return True

    except Exception as e:
        logger.error(f"Nuclear replace error: {e}")
        return False
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _mix_audio(video_path: str, audio_path: str, output_path: str,
               original_volume: float, commentary_volume: float) -> bool:
    """
    Mix commentary over original audio with volume ducking.
    Original audio is reduced to `original_volume`, commentary at `commentary_volume`.
    """
    # Build amix filter with volume adjustments
    # Boost commentary by 1.5x, add limiter, normalize=0 prevents amix halving
    comm_vol = max(commentary_volume, 1.0) * 1.5
    filter_complex = (
        f"[0:a]volume={original_volume}[orig];"
        f"[1:a]volume={comm_vol},alimiter=limit=0.95[comm];"
        f"[orig][comm]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,alimiter=limit=0.95[aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
        "-shortest",
        "-movflags", "+faststart",
        output_path
    ]

    result = subprocess.run(
        cmd, capture_output=True, encoding='utf-8',
        errors='replace', timeout=180
    )

    if result.returncode != 0:
        logger.error(f"Mix audio failed: {result.stderr[-300:]}")
        # Fallback: try simpler approach without filter_complex
        return _mix_audio_simple(video_path, audio_path, output_path,
                                 original_volume, commentary_volume)
    return True


def _mix_audio_simple(video_path: str, audio_path: str, output_path: str,
                       original_volume: float, commentary_volume: float) -> bool:
    """Simpler mix fallback — extract, adjust, combine."""
    import tempfile

    tmp_dir = tempfile.mkdtemp()
    orig_adj = os.path.join(tmp_dir, "orig_adj.aac")
    comm_adj = os.path.join(tmp_dir, "comm_adj.aac")

    try:
        # Extract and adjust original audio
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-af", f"volume={original_volume}",
            "-c:a", "aac", "-b:a", "256k", orig_adj
        ], capture_output=True, encoding='utf-8', errors='replace', timeout=60)

        # Adjust commentary volume
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-af", f"volume={commentary_volume}",
            "-c:a", "aac", "-b:a", "256k", comm_adj
        ], capture_output=True, encoding='utf-8', errors='replace', timeout=60)

        # Merge
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", orig_adj,
            "-i", comm_adj,
            "-filter_complex", "[1:a][2:a]amix=inputs=2:duration=first[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            output_path
        ], capture_output=True, encoding='utf-8', errors='replace', timeout=120)

        return result.returncode == 0

    except Exception as e:
        logger.error(f"Simple mix fallback failed: {e}")
        return False
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _get_duration(path: str) -> float:
    """Get media file duration via ffprobe."""
    if not os.path.isfile(path):
        return 0.0
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", path
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        info = json.loads(r.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0
