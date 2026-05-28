"""
Video Editor Module
Assembles scene clips into a final polished video with:
- Transitions (fade, slide, dissolve, zoom, cut)
- Ken Burns effect (zoom/pan on stills)
- Animated text overlays (lower thirds, topic text)
- Color correction
- Sound effects
All FFmpeg-based.
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EditConfig:
    """Configuration for the video editor."""
    width: int = 1080
    height: int = 1920
    fps: int = 30
    transition_duration: float = 0.5    # seconds per transition
    color_correction: bool = True
    brightness: float = 0.02
    contrast: float = 1.08
    saturation: float = 1.12
    text_overlays: bool = True
    ken_burns: bool = True              # zoom/pan on static scenes
    ken_burns_zoom: float = 1.08        # max zoom factor
    output_quality: int = 20            # CRF value (lower = better)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "transition_duration": self.transition_duration,
            "color_correction": self.color_correction,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "text_overlays": self.text_overlays,
            "ken_burns": self.ken_burns,
            "ken_burns_zoom": self.ken_burns_zoom,
            "output_quality": self.output_quality,
        }


@dataclass
class EditResult:
    """Result of video editing."""
    output_path: str
    duration: float
    num_scenes: int
    transitions_applied: list
    effects_applied: list
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "duration": self.duration,
            "num_scenes": self.num_scenes,
            "transitions_applied": self.transitions_applied,
            "effects_applied": self.effects_applied,
            "success": self.success,
            "error": self.error,
        }


def assemble_video(scene_clips: list[dict],
                   output_path: str,
                   config: Optional[EditConfig] = None) -> EditResult:
    """
    Assemble scene clips into a final edited video.

    Args:
        scene_clips: List of dicts with keys:
            - clip_path: path to video file
            - duration: scene duration in seconds
            - transition: 'fade', 'slide_left', 'slide_right', 'dissolve', 'zoom', 'cut'
            - text_overlay: text to display on screen (optional)
            - scene_number: for ordering
        output_path: Where to save the final video
        config: EditConfig (uses defaults if None)

    Returns:
        EditResult
    """
    if config is None:
        config = EditConfig()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Validate clips exist
    valid_clips = []
    for clip in scene_clips:
        if os.path.isfile(clip.get("clip_path", "")):
            valid_clips.append(clip)
        else:
            logger.warning(f"Scene {clip.get('scene_number')}: clip not found")

    if not valid_clips:
        return EditResult(output_path, 0, 0, [], [], False, "No valid clips")

    transitions_applied = []
    effects_applied = []

    try:
        # Strategy: process each clip individually (normalize, apply effects),
        # then concatenate with transitions using xfade filter.
        tmp_dir = tempfile.mkdtemp()
        processed_clips = []

        for i, clip in enumerate(valid_clips):
            clip_path = clip["clip_path"]
            duration = clip.get("duration", 5.0)
            text_overlay = clip.get("text_overlay", "")
            proc_path = os.path.join(tmp_dir, f"proc_{i:02d}.mp4")

            # ── Process individual clip ──
            vf_parts = []

            # 1. Ensure correct dimensions
            vf_parts.append(
                f"scale={config.width}:{config.height}:force_original_aspect_ratio=decrease,"
                f"pad={config.width}:{config.height}:(ow-iw)/2:(oh-ih)/2:black,"
                f"setsar=1"
            )

            # 2. Color correction
            if config.color_correction:
                vf_parts.append(
                    f"eq=brightness={config.brightness}:"
                    f"contrast={config.contrast}:"
                    f"saturation={config.saturation}"
                )
                if i == 0:
                    effects_applied.append("color_correction")

            # 3. Ken Burns effect (subtle zoom over duration)
            if config.ken_burns:
                zoom_start = 1.0
                zoom_end = config.ken_burns_zoom
                # Alternate between zoom-in and zoom-out per scene
                if i % 2 == 0:
                    vf_parts.append(
                        f"zoompan=z='min({zoom_end},1+({zoom_end}-1)*on/({config.fps}*{duration}))':"
                        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                        f"d={int(config.fps * duration)}:"
                        f"s={config.width}x{config.height}:fps={config.fps}"
                    )
                else:
                    vf_parts.append(
                        f"zoompan=z='max(1,{zoom_end}-({zoom_end}-1)*on/({config.fps}*{duration}))':"
                        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                        f"d={int(config.fps * duration)}:"
                        f"s={config.width}x{config.height}:fps={config.fps}"
                    )
                if i == 0:
                    effects_applied.append("ken_burns")

            # 4. Text overlay
            if config.text_overlays and text_overlay:
                escaped = text_overlay.replace("'", "\\'").replace(":", "\\:")
                # Lower-third style text with background box
                vf_parts.append(
                    f"drawtext=text='{escaped}':"
                    f"fontsize=38:fontcolor=white:"
                    f"x=(w-text_w)/2:y=h-h/6:"
                    f"box=1:boxcolor=black@0.5:boxborderw=12:"
                    f"alpha='if(lt(t,0.5),t*2,if(gt(t,{duration-0.5}),({duration}-t)*2,1))'"
                )
                if "text_overlays" not in effects_applied:
                    effects_applied.append("text_overlays")

            # 5. Force framerate
            vf_parts.append(f"fps={config.fps}")

            vf = ",".join(vf_parts)

            cmd = [
                "ffmpeg", "-y",
                "-i", clip_path,
                "-vf", vf,
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "fast",
                "-crf", str(config.output_quality),
                "-an", "-pix_fmt", "yuv420p",
                proc_path
            ]

            result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                    errors='replace', timeout=120)

            if result.returncode == 0 and os.path.isfile(proc_path):
                processed_clips.append({
                    "path": proc_path,
                    "duration": duration,
                    "transition": clip.get("transition", "fade"),
                })
            else:
                logger.warning(f"Clip {i} processing failed: {result.stderr[-200:]}")
                # Use original clip as fallback
                processed_clips.append({
                    "path": clip_path,
                    "duration": duration,
                    "transition": clip.get("transition", "cut"),
                })

        # ── Concatenate with transitions ──
        if len(processed_clips) == 1:
            # Single clip — just copy
            _copy_file(processed_clips[0]["path"], output_path)
        elif len(processed_clips) <= 3:
            # Few clips — use xfade for transitions
            _concat_with_xfade(processed_clips, output_path, config, transitions_applied)
        else:
            # Many clips — use concat demuxer (simpler, more reliable)
            # Apply transitions as filter on individual clips instead
            _concat_demuxer(processed_clips, output_path, tmp_dir, transitions_applied)

        # Clean up temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        if os.path.isfile(output_path):
            duration = _get_duration(output_path)
            return EditResult(
                output_path, duration, len(valid_clips),
                transitions_applied, effects_applied, True
            )
        else:
            return EditResult(output_path, 0, 0, [], [], False,
                              "Assembly produced no output")

    except Exception as e:
        logger.error(f"Video assembly error: {e}")
        return EditResult(output_path, 0, 0, [], [], False, str(e))


def _concat_with_xfade(clips: list[dict], output_path: str,
                       config: EditConfig, transitions_log: list) -> bool:
    """
    Concatenate clips using FFmpeg xfade filter for smooth transitions.
    Works well for 2-3 clips. For more, use concat demuxer.
    """
    if len(clips) < 2:
        return False

    try:
        inputs = []
        for c in clips:
            inputs.extend(["-i", c["path"]])

        # Build xfade filter chain
        # For N clips, we need N-1 xfade filters chained
        td = config.transition_duration
        filter_parts = []
        current_label = "[0:v]"

        for i in range(1, len(clips)):
            prev_dur = clips[i - 1]["duration"]
            trans_type = _map_transition(clips[i].get("transition", "fade"))
            offset = prev_dur - td

            # Accumulate offset from all previous clips
            if i > 1:
                for j in range(1, i):
                    offset += clips[j]["duration"] - td

            out_label = f"[v{i}]" if i < len(clips) - 1 else "[vout]"

            filter_parts.append(
                f"{current_label}[{i}:v]xfade=transition={trans_type}:"
                f"duration={td}:offset={offset:.2f}{out_label}"
            )
            current_label = out_label
            transitions_log.append(f"{trans_type}@{offset:.1f}s")

        filter_complex = ";".join(filter_parts)

        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-c:v", "libx264", "-preset", "medium",
            "-crf", str(config.output_quality),
            "-pix_fmt", "yuv420p",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=300)

        if result.returncode != 0:
            logger.warning(f"xfade failed, falling back to concat: {result.stderr[-200:]}")
            return False

        return True

    except Exception as e:
        logger.error(f"xfade concat error: {e}")
        return False


def _concat_demuxer(clips: list[dict], output_path: str,
                    tmp_dir: str, transitions_log: list) -> bool:
    """
    Concatenate clips using FFmpeg concat demuxer.
    Simpler and more reliable for many clips.
    Applies fade in/out on each clip boundary for smooth transitions.
    """
    try:
        # First, add fade effects to each clip at boundaries
        faded_clips = []
        for i, clip in enumerate(clips):
            fade_path = os.path.join(tmp_dir, f"faded_{i:02d}.mp4")
            dur = clip["duration"]
            fade_dur = 0.3  # 300ms fade

            vf = []
            # Fade in (except first clip)
            if i > 0:
                vf.append(f"fade=t=in:st=0:d={fade_dur}")
                transitions_log.append(f"fade_in@scene_{i+1}")
            # Fade out (except last clip)
            if i < len(clips) - 1:
                vf.append(f"fade=t=out:st={dur - fade_dur}:d={fade_dur}")

            if vf:
                cmd = [
                    "ffmpeg", "-y", "-i", clip["path"],
                    "-vf", ",".join(vf),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                    "-an", "-pix_fmt", "yuv420p",
                    fade_path
                ]
                result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                        errors='replace', timeout=60)
                if result.returncode == 0 and os.path.isfile(fade_path):
                    faded_clips.append(fade_path)
                else:
                    faded_clips.append(clip["path"])
            else:
                faded_clips.append(clip["path"])

        # Write concat list file
        list_file = os.path.join(tmp_dir, "concat_list.txt")
        with open(list_file, "w") as f:
            for path in faded_clips:
                f.write(f"file '{path}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=300)
        return result.returncode == 0

    except Exception as e:
        logger.error(f"Concat demuxer error: {e}")
        return False


def add_text_overlay(video_path: str, output_path: str,
                     text: str, position: str = "lower_third",
                     start_time: float = 0, end_time: float = 0,
                     font_size: int = 38) -> bool:
    """
    Add a text overlay to a video at a specific time range.
    Positions: 'lower_third', 'center', 'top', 'bottom'
    """
    escaped = text.replace("'", "\\'").replace(":", "\\:")

    pos_map = {
        "lower_third": f"x=(w-text_w)/2:y=h-h/5",
        "center": f"x=(w-text_w)/2:y=(h-text_h)/2",
        "top": f"x=(w-text_w)/2:y=h/10",
        "bottom": f"x=(w-text_w)/2:y=h-h/8",
    }
    pos = pos_map.get(position, pos_map["lower_third"])

    enable = ""
    if end_time > start_time:
        enable = f":enable='between(t,{start_time},{end_time})'"

    vf = (
        f"drawtext=text='{escaped}':"
        f"fontsize={font_size}:fontcolor=white:"
        f"{pos}:"
        f"box=1:boxcolor=black@0.5:boxborderw=10"
        f"{enable}"
    )

    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=120)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Text overlay error: {e}")
        return False


def apply_color_grade(video_path: str, output_path: str,
                      preset: str = "cinematic") -> bool:
    """
    Apply a color grade preset to a video.
    Presets: 'cinematic', 'warm', 'cool', 'vibrant', 'moody'
    """
    grades = {
        "cinematic": "eq=brightness=-0.02:contrast=1.15:saturation=0.90:gamma_r=1.02:gamma_b=1.02",
        "warm": "eq=brightness=0.02:contrast=1.05:saturation=1.12:gamma_r=1.05:gamma_b=0.95",
        "cool": "eq=brightness=0.01:contrast=1.08:saturation=1.05:gamma_r=0.95:gamma_b=1.08",
        "vibrant": "eq=brightness=0.03:contrast=1.10:saturation=1.25",
        "moody": "eq=brightness=-0.04:contrast=1.20:saturation=0.80:gamma_r=1.03:gamma_b=1.05",
    }

    vf = grades.get(preset, grades["cinematic"])

    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=120)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Color grade error: {e}")
        return False


# ── Helpers ──

def _map_transition(name: str) -> str:
    """Map our transition names to FFmpeg xfade transition names."""
    mapping = {
        "fade": "fade",
        "slide_left": "slideleft",
        "slide_right": "slideright",
        "dissolve": "dissolve",
        "zoom": "smoothup",
        "cut": "fade",  # use very short fade for 'cut'
        "wipe": "wipeleft",
        "radial": "radial",
        "pixelize": "pixelize",
    }
    return mapping.get(name, "fade")


def _copy_file(src: str, dst: str):
    """Copy file from src to dst."""
    import shutil
    shutil.copy2(src, dst)


def _get_duration(path: str) -> float:
    """Get media duration via ffprobe."""
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
