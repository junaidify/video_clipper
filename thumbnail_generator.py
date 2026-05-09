"""
Thumbnail Generator Module
Extracts best frames from video clips and generates YouTube-style thumbnails.
Two modes: template-based (local) and AI-powered (Gemini).
"""

import base64
import json
import logging
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_font_path() -> str:
    """Get a valid bold font path for the current OS."""
    if platform.system() == "Windows":
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    for p in candidates:
        if os.path.isfile(p):
            return p.replace("\\", "/")
    return ""


def _font_param() -> str:
    """Return FFmpeg fontfile or font param string."""
    fp = _get_font_path()
    if fp:
        escaped = fp.replace(":", "\\:")
        return f"fontfile='{escaped}'"
    return "font=Arial"


def _get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return 0.0


def _get_video_dimensions(video_path: str) -> tuple:
    """Get video width and height."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception:
        return 1080, 1920


def extract_frames(video_path: str, num_frames: int = 10) -> list:
    """
    Extract evenly-spaced frames from video for analysis.

    Returns:
        List of temp file paths to extracted frames (PNG).
    """
    duration = _get_video_duration(video_path)
    if duration <= 0:
        duration = 10.0

    frame_paths = []
    interval = duration / (num_frames + 1)

    for i in range(1, num_frames + 1):
        timestamp = interval * i
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            frame_path = tmp.name

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            frame_path
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        if result.returncode == 0 and os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
            frame_paths.append({"path": frame_path, "timestamp": timestamp})
        else:
            if os.path.exists(frame_path):
                os.unlink(frame_path)

    return frame_paths


def pick_best_frame(video_path: str) -> str:
    """
    Extract the best frame from video based on visual quality metrics.
    Uses FFmpeg's thumbnail filter which picks high-contrast representative frames.

    Returns:
        Path to best frame PNG.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        frame_path = tmp.name

    # FFmpeg thumbnail filter picks visually interesting frames
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", "thumbnail=300,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black",
        "-vframes", "1",
        "-q:v", "2",
        frame_path
    ]
    result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')

    if result.returncode != 0 or not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
        # Fallback: grab frame at 30% mark
        duration = _get_video_duration(video_path)
        ts = duration * 0.3 if duration > 0 else 2.0
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", video_path,
            "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black",
            "-vframes", "1",
            "-q:v", "2",
            frame_path
        ]
        subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')

    return frame_path


def generate_template_thumbnail(video_path: str, title: str, output_path: str,
                                 style: str = "bold") -> str:
    """
    Generate a YouTube-style thumbnail using FFmpeg filters.
    Applies: best frame + brightness/contrast boost + bold text overlay + vignette.

    Args:
        video_path: Input video clip
        title: Text to overlay on thumbnail
        output_path: Where to save the thumbnail PNG
        style: "bold" (high contrast) or "clean" (minimal)

    Returns:
        Path to generated thumbnail
    """
    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())

    # Extract best frame first
    best_frame = pick_best_frame(video_path)

    try:
        # Escape text for FFmpeg
        safe_title = title.upper().replace("'", "\\'").replace(":", "\\:").replace("%", "%%")

        if style == "bold":
            # YouTube-style: high contrast, vignette, large bold text
            vf_parts = [
                # Boost contrast and saturation
                "eq=contrast=1.3:brightness=0.05:saturation=1.4",
                # Vignette effect
                "vignette=PI/4",
                # Large bold title text - centered
                f"drawtext=text='{safe_title}':"
                f"x=(w-text_w)/2:y=(h-text_h)/2:"
                f"fontsize=64:fontcolor=white:"
                f"borderw=5:bordercolor=black:"
                f"shadowx=3:shadowy=3:shadowcolor=black@0.6:"
                f"{_font_param()}",
            ]
        else:
            # Clean style: subtle enhancement, bottom text
            vf_parts = [
                "eq=contrast=1.15:brightness=0.03:saturation=1.2",
                # Semi-transparent bottom bar
                f"drawbox=x=0:y=ih-ih/4:w=iw:h=ih/4:color=black@0.6:t=fill",
                f"drawtext=text='{safe_title}':"
                f"x=(w-text_w)/2:y=h-h/4+(h/4-text_h)/2:"
                f"fontsize=48:fontcolor=white:"
                f"borderw=2:bordercolor=black:"
                f"{_font_param()}",
            ]

        vf = ",".join(vf_parts)

        cmd = [
            "ffmpeg", "-y",
            "-i", best_frame,
            "-vf", vf,
            "-q:v", "2",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        if result.returncode != 0:
            logger.error(f"Template thumbnail failed: {result.stderr}")
            raise RuntimeError(f"FFmpeg error: {result.stderr[-500:]}")

        logger.info(f"Template thumbnail saved: {output_path}")
        return output_path
    finally:
        if os.path.exists(best_frame):
            os.unlink(best_frame)


def generate_ai_thumbnail(video_path: str, title: str, output_path: str,
                           context: str = "") -> str:
    """
    Generate a thumbnail using Gemini AI to pick the best frame
    and suggest text placement / styling.

    Process:
    1. Extract candidate frames
    2. Send to Gemini to pick best frame + suggest overlay text/position
    3. Apply Gemini's recommendations via FFmpeg

    Args:
        video_path: Input video clip
        title: Video/clip title
        output_path: Where to save the thumbnail
        context: Optional context about the video content

    Returns:
        Path to generated thumbnail
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY required for AI thumbnail generation")

    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Install google-generativeai: pip install google-generativeai")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Extract candidate frames
    frames = extract_frames(video_path, num_frames=6)
    if not frames:
        raise RuntimeError("Could not extract frames from video")

    try:
        # Send frames to Gemini for analysis
        prompt_parts = [
            f"""You are a YouTube thumbnail design expert. Analyze these {len(frames)} frames from a video clip titled "{title}".
{f'Context: {context}' if context else ''}

Pick the BEST frame for a thumbnail (the one that would get the most clicks - look for expressive faces, action, emotion, or visually striking composition).

Then suggest a short, punchy headline (max 5 words) that would make people click, inspired by top YouTube channels' thumbnail style.

Return ONLY a JSON object (no markdown fences):
{{
    "best_frame_index": 0,
    "headline": "YOUR PUNCHY HEADLINE",
    "headline_position": "center",
    "style": "bold",
    "text_color": "#FFFFFF",
    "reasoning": "brief explanation"
}}

headline_position can be: "center", "top", "bottom"
style can be: "bold" (high contrast, large text) or "clean" (subtle, elegant)"""
        ]

        # Add frame images
        for i, frame in enumerate(frames):
            with open(frame["path"], "rb") as f:
                img_data = f.read()
            prompt_parts.append({
                "mime_type": "image/png",
                "data": base64.b64encode(img_data).decode()
            })
            prompt_parts.append(f"Frame {i} (at {frame['timestamp']:.1f}s)")

        response = model.generate_content(prompt_parts)
        text = response.text.strip()

        # Clean markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        suggestion = json.loads(text)
        logger.info(f"Gemini suggestion: {suggestion}")

        best_idx = suggestion.get("best_frame_index", 0)
        headline = suggestion.get("headline", title[:30])
        position = suggestion.get("headline_position", "center")
        style = suggestion.get("style", "bold")
        text_color = suggestion.get("text_color", "#FFFFFF")

        # Clamp index
        best_idx = max(0, min(best_idx, len(frames) - 1))
        best_frame_path = frames[best_idx]["path"]

        # Apply styling to the best frame
        safe_headline = headline.upper().replace("'", "\\'").replace(":", "\\:").replace("%", "%%")

        # Position mapping
        if position == "top":
            y_expr = "h*0.15"
        elif position == "bottom":
            y_expr = "h*0.75"
        else:
            y_expr = "(h-text_h)/2"

        # Scale frame to 1280x720
        vf_parts = [
            "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black",
        ]

        if style == "bold":
            vf_parts.extend([
                "eq=contrast=1.35:brightness=0.05:saturation=1.5",
                "vignette=PI/4",
                f"drawtext=text='{safe_headline}':"
                f"x=(w-text_w)/2:y={y_expr}:"
                f"fontsize=72:fontcolor={text_color}:"
                f"borderw=6:bordercolor=black:"
                f"shadowx=4:shadowy=4:shadowcolor=black@0.7:"
                f"{_font_param()}",
            ])
        else:
            vf_parts.extend([
                "eq=contrast=1.15:brightness=0.03:saturation=1.2",
                f"drawtext=text='{safe_headline}':"
                f"x=(w-text_w)/2:y={y_expr}:"
                f"fontsize=56:fontcolor={text_color}:"
                f"borderw=3:bordercolor=black:"
                f"{_font_param()}",
            ])

        vf = ",".join(vf_parts)

        cmd = [
            "ffmpeg", "-y",
            "-i", best_frame_path,
            "-vf", vf,
            "-q:v", "2",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        if result.returncode != 0:
            logger.error(f"AI thumbnail FFmpeg failed: {result.stderr}")
            raise RuntimeError(f"FFmpeg error: {result.stderr[-500:]}")

        logger.info(f"AI thumbnail saved: {output_path}")
        return output_path

    finally:
        # Clean up extracted frames
        for frame in frames:
            if os.path.exists(frame["path"]):
                os.unlink(frame["path"])
