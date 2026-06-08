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
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _ffmpeg_font_setup() -> dict:
    """
    Set up a portable font for FFmpeg drawtext on Windows.
    Copies a system font to a temp dir so we can reference it without C: colon.
    Returns dict with 'env', 'font_param', and 'cwd'.
    """
    setup_dir = os.path.join(tempfile.gettempdir(), "ffmpeg_fonts")
    os.makedirs(setup_dir, exist_ok=True)

    env = os.environ.copy()

    if platform.system() == "Windows":
        fc_conf = os.path.join(setup_dir, "fonts.conf")
        fonts_dir_fwd = "C:/Windows/Fonts"
        cache_dir_fwd = setup_dir.replace("\\", "/")
        if not os.path.isfile(fc_conf):
            with open(fc_conf, "w", encoding="utf-8") as f:
                f.write(
                    '<?xml version="1.0"?>\n'
                    '<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">\n'
                    '<fontconfig>\n'
                    f'  <dir>{fonts_dir_fwd}</dir>\n'
                    f'  <cachedir>{cache_dir_fwd}</cachedir>\n'
                    '</fontconfig>\n'
                )
        env["FONTCONFIG_FILE"] = fc_conf
        env["FONTCONFIG_PATH"] = setup_dir

        font_dst = os.path.join(setup_dir, "font.ttf")
        if not os.path.isfile(font_dst):
            for src in [
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
            ]:
                if os.path.isfile(src):
                    shutil.copy2(src, font_dst)
                    break

        font_param = "fontfile=font.ttf" if os.path.isfile(font_dst) else "font=Arial"
    else:
        for p in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]:
            if os.path.isfile(p):
                font_param = f"fontfile={p}"
                break
        else:
            font_param = "font=Arial"

    return {"env": env, "font_param": font_param, "cwd": setup_dir}


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


def score_frame(frame_path: str) -> float:
    """
    Score a single frame for visual impact using FFmpeg signalstats.
    Higher score = more visually striking = better thumbnail candidate.

    Scores based on:
    - Contrast (standard deviation of luminance — high = dynamic range)
    - Saturation (colorfulness — high = eye-catching)
    - Entropy approximation (file size as proxy — complex scenes score higher)
    """
    score = 0.0

    try:
        # Use FFmpeg signalstats to get YMIN, YMAX, SATMIN, SATMAX
        cmd = [
            "ffprobe", "-v", "quiet",
            "-f", "lavfi",
            "-i", f"movie='{frame_path.replace(os.sep, '/')}',signalstats",
            "-show_entries", "frame_tags=lavfi.signalstats.YMIN,lavfi.signalstats.YMAX,"
                            "lavfi.signalstats.YAVG,lavfi.signalstats.SATAVG",
            "-print_format", "json",
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=10)

        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            frames = data.get("frames", [])
            if frames:
                tags = frames[0].get("tags", {})
                ymin = float(tags.get("lavfi.signalstats.YMIN", 16))
                ymax = float(tags.get("lavfi.signalstats.YMAX", 235))
                yavg = float(tags.get("lavfi.signalstats.YAVG", 128))
                satavg = float(tags.get("lavfi.signalstats.SATAVG", 50))

                # Contrast score: dynamic range of luminance (0-255)
                contrast = (ymax - ymin) / 255.0
                score += contrast * 40  # max 40 points

                # Brightness penalty: avoid too dark or too bright
                brightness_deviation = abs(yavg - 128) / 128.0
                score -= brightness_deviation * 10  # penalty up to 10

                # Saturation score: colorful frames are more eye-catching
                sat_score = min(satavg / 100.0, 1.0)
                score += sat_score * 30  # max 30 points
    except Exception:
        pass

    # File size as entropy proxy: complex/detailed frames = larger files
    try:
        fsize = os.path.getsize(frame_path)
        # Normalize: typical PNG frame is 200KB-2MB
        size_score = min(fsize / (1024 * 1024), 1.0)  # cap at 1MB
        score += size_score * 20  # max 20 points
    except Exception:
        pass

    return max(score, 0.0)


def _detect_scene_changes(video_path: str, threshold: float = 0.3) -> list:
    """
    Detect scene-change timestamps using FFmpeg's scene filter.
    Scene changes = visual peaks where something dramatic happens.
    Returns list of timestamps (float seconds).
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8',
            errors='replace', timeout=60
        )
        import re
        timestamps = []
        for match in re.finditer(r'pts_time:(\d+\.?\d*)', result.stderr):
            ts = float(match.group(1))
            timestamps.append(ts)
        logger.info(f"Scene changes detected: {len(timestamps)} at threshold={threshold}")
        return timestamps
    except Exception as e:
        logger.warning(f"Scene detection failed: {e}")
        return []


def _extract_frame_at(video_path: str, timestamp: float, label: str = "") -> Optional[dict]:
    """Extract a single frame at a specific timestamp. Returns dict or None."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        frame_path = tmp.name

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(max(0, timestamp)),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        frame_path
    ]
    result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
    if result.returncode == 0 and os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
        return {"path": frame_path, "timestamp": round(timestamp, 3), "label": label}
    else:
        if os.path.exists(frame_path):
            os.unlink(frame_path)
        return None


def pick_top_frames(video_path: str, num_candidates: int = 20,
                     top_n: int = 5) -> list:
    """
    Find the most eye-catching frames from ANYWHERE in the video.

    Multi-strategy extraction:
    1. First frame + last frame (always included as candidates)
    2. Scene-change detection (dramatic visual shifts = action peaks)
    3. FFmpeg thumbnail filter (built-in visual-interest detection)
    4. Evenly-spaced grid (fallback to cover gaps)

    All candidates scored by contrast, saturation, and visual complexity.

    Returns:
        List of dicts: [{"path": str, "timestamp": float, "score": float}, ...]
        Sorted by score descending (best first).
    """
    duration = _get_video_duration(video_path)
    if duration <= 0:
        duration = 10.0

    candidates = []
    seen_timestamps = set()  # avoid duplicate timestamps within 0.3s

    def _is_duplicate(ts):
        for s in seen_timestamps:
            if abs(s - ts) < 0.3:
                return True
        return False

    def _add_candidate(frame_dict):
        if frame_dict and not _is_duplicate(frame_dict["timestamp"]):
            seen_timestamps.add(frame_dict["timestamp"])
            candidates.append(frame_dict)

    # ── Strategy 1: First frame + last frame ──
    # These are often the most impactful (opening action, closing climax)
    _add_candidate(_extract_frame_at(video_path, 0.1, "first"))
    _add_candidate(_extract_frame_at(video_path, max(0.5, duration - 0.3), "last"))
    # Also grab near-end (0.5s before end) for the "WTF frame"
    if duration > 2:
        _add_candidate(_extract_frame_at(video_path, duration - 1.0, "near_end"))

    # ── Strategy 2: Scene-change detection ──
    # Find moments where the visual content shifts dramatically
    scene_timestamps = _detect_scene_changes(video_path, threshold=0.25)
    for ts in scene_timestamps[:8]:  # cap at 8 scene changes
        _add_candidate(_extract_frame_at(video_path, ts, "scene_change"))

    # ── Strategy 3: FFmpeg thumbnail filter ──
    # FFmpeg's built-in algorithm picks high-contrast, visually interesting frames
    for batch in range(min(3, max(1, int(duration / 3)))):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            thumb_path = tmp.name
        # thumbnail=N means analyze N frames and pick the best one
        skip_frames = batch * 100
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"select='gte(n,{skip_frames})',thumbnail=100",
            "-vframes", "1",
            "-q:v", "2",
            thumb_path
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=30)
        if result.returncode == 0 and os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0:
            # Estimate timestamp from frame number (approximate)
            est_ts = (skip_frames + 50) / 24.0  # assuming ~24fps
            est_ts = min(est_ts, duration - 0.1)
            frame = {"path": thumb_path, "timestamp": round(est_ts, 3), "label": "ffmpeg_pick"}
            _add_candidate(frame)
        else:
            if os.path.exists(thumb_path):
                os.unlink(thumb_path)

    # ── Strategy 4: Evenly-spaced grid (fill remaining slots) ──
    grid_count = max(4, num_candidates - len(candidates))
    interval = duration / (grid_count + 1)
    for i in range(1, grid_count + 1):
        ts = interval * i
        if not _is_duplicate(ts):
            _add_candidate(_extract_frame_at(video_path, ts, "grid"))

    # ── Score all candidates ──
    scored = []
    for frame in candidates:
        frame["score"] = round(score_frame(frame["path"]), 2)
        scored.append(frame)

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Clean up frames that didn't make the cut
    for frame in scored[top_n:]:
        if os.path.exists(frame["path"]):
            os.unlink(frame["path"])

    top = scored[:top_n]
    labels = [f["label"] for f in top]
    logger.info(
        f"Thumbnail candidates: {len(candidates)} extracted, "
        f"top {len(top)} scored [{', '.join(labels)}], "
        f"best={top[0]['score'] if top else 0}"
    )
    return top


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
        font_setup = _ffmpeg_font_setup()

        def _esc(txt):
            for ch in ("\\", "'", ":", ";", "[", "]", ","):
                txt = txt.replace(ch, "\\" + ch)
            txt = txt.replace("%", "%%")
            return txt

        safe_title = _esc(title.upper())

        if style == "bold":
            vf_parts = [
                "eq=contrast=1.3:brightness=0.05:saturation=1.4",
                "vignette=PI/4",
                f"drawtext=text='{safe_title}':"
                f"x=(w-text_w)/2:y=(h-text_h)/2:"
                f"fontsize=64:fontcolor=white:"
                f"borderw=5:bordercolor=black:"
                f"shadowx=3:shadowy=3:shadowcolor=black@0.6:"
                f"{font_setup['font_param']}",
            ]
        else:
            vf_parts = [
                "eq=contrast=1.15:brightness=0.03:saturation=1.2",
                f"drawbox=x=0:y=ih-ih/4:w=iw:h=ih/4:color=black@0.6:t=fill",
                f"drawtext=text='{safe_title}':"
                f"x=(w-text_w)/2:y=h-h/4+(h/4-text_h)/2:"
                f"fontsize=48:fontcolor=white:"
                f"borderw=2:bordercolor=black:"
                f"{font_setup['font_param']}",
            ]

        vf = ",".join(vf_parts)

        cmd = [
            "ffmpeg", "-y",
            "-i", best_frame,
            "-vf", vf,
            "-q:v", "2",
            output_path
        ]

        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8', errors='replace',
            env=font_setup["env"],
            cwd=font_setup["cwd"],
        )
        if result.returncode != 0:
            logger.error(f"Template thumbnail FULL stderr:\n{result.stderr}")
            raise RuntimeError(f"Template thumbnail failed (code {result.returncode}). "
                               f"Check server logs.")

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

        font_setup = _ffmpeg_font_setup()

        def _esc(txt):
            for ch in ("\\", "'", ":", ";", "[", "]", ","):
                txt = txt.replace(ch, "\\" + ch)
            txt = txt.replace("%", "%%")
            return txt

        safe_headline = _esc(headline.upper())

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
                f"fontsize=72:fontcolor='{text_color}':"
                f"borderw=6:bordercolor=black:"
                f"shadowx=4:shadowy=4:shadowcolor=black@0.7:"
                f"{font_setup['font_param']}",
            ])
        else:
            vf_parts.extend([
                "eq=contrast=1.15:brightness=0.03:saturation=1.2",
                f"drawtext=text='{safe_headline}':"
                f"x=(w-text_w)/2:y={y_expr}:"
                f"fontsize=56:fontcolor='{text_color}':"
                f"borderw=3:bordercolor=black:"
                f"{font_setup['font_param']}",
            ])

        vf = ",".join(vf_parts)

        cmd = [
            "ffmpeg", "-y",
            "-i", best_frame_path,
            "-vf", vf,
            "-q:v", "2",
            output_path
        ]

        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8', errors='replace',
            env=font_setup["env"],
            cwd=font_setup["cwd"],
        )
        if result.returncode != 0:
            logger.error(f"AI thumbnail FULL stderr:\n{result.stderr}")
            raise RuntimeError(f"AI thumbnail failed (code {result.returncode}). "
                               f"Check server logs.")

        logger.info(f"AI thumbnail saved: {output_path}")
        return output_path

    finally:
        # Clean up extracted frames
        for frame in frames:
            if os.path.exists(frame["path"]):
                os.unlink(frame["path"])
