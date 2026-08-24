"""
Thumbnail Generator Module
Extracts visual quality peaks and generates click-worthy YouTube/Shorts thumbnails.
Supports template-based local processing and AI-assisted (Gemini) headline & composition.
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
from typing import Optional, List

logger = logging.getLogger(__name__)


def _ffmpeg_font_setup() -> dict:
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


def score_frame(frame_path: str) -> float:
    """
    Score a single frame for visual impact using FFmpeg signalstats.
    Evaluates contrast, saturation, and file complexity.
    """
    score = 0.0

    try:
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

                contrast = (ymax - ymin) / 255.0
                score += contrast * 40

                brightness_deviation = abs(yavg - 128) / 128.0
                score -= brightness_deviation * 10

                sat_score = min(satavg / 100.0, 1.0)
                score += sat_score * 30
    except Exception:
        pass

    try:
        fsize = os.path.getsize(frame_path)
        size_score = min(fsize / (1024 * 1024), 1.0)
        score += size_score * 20
    except Exception:
        pass

    return max(score, 0.0)


def _detect_scene_changes(video_path: str, threshold: float = 0.3) -> List[float]:
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
        return timestamps
    except Exception:
        return []


def _extract_frame_at(video_path: str, timestamp: float, label: str = "") -> Optional[dict]:
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


def pick_top_frames(video_path: str, num_candidates: int = 20, top_n: int = 5) -> List[dict]:
    """Find the most eye-catching candidate frames across the video."""
    duration = _get_video_duration(video_path)
    if duration <= 0:
        duration = 10.0

    candidates = []
    seen_timestamps = set()

    def _is_duplicate(ts):
        for s in seen_timestamps:
            if abs(s - ts) < 0.3:
                return True
        return False

    def _add_candidate(frame_dict):
        if frame_dict and not _is_duplicate(frame_dict["timestamp"]):
            seen_timestamps.add(frame_dict["timestamp"])
            candidates.append(frame_dict)

    _add_candidate(_extract_frame_at(video_path, 0.1, "first"))
    _add_candidate(_extract_frame_at(video_path, max(0.5, duration - 0.3), "last"))
    if duration > 2:
        _add_candidate(_extract_frame_at(video_path, duration - 1.0, "near_end"))

    scene_timestamps = _detect_scene_changes(video_path, threshold=0.25)
    for ts in scene_timestamps[:8]:
        _add_candidate(_extract_frame_at(video_path, ts, "scene_change"))

    grid_count = max(4, num_candidates - len(candidates))
    interval = duration / (grid_count + 1)
    for i in range(1, grid_count + 1):
        ts = interval * i
        if not _is_duplicate(ts):
            _add_candidate(_extract_frame_at(video_path, ts, "grid"))

    scored = []
    for frame in candidates:
        frame["score"] = round(score_frame(frame["path"]), 2)
        scored.append(frame)

    scored.sort(key=lambda x: x["score"], reverse=True)

    for frame in scored[top_n:]:
        if os.path.exists(frame["path"]):
            os.unlink(frame["path"])

    return scored[:top_n]


def extract_frames(video_path: str, num_frames: int = 10) -> List[dict]:
    """Extract evenly-spaced frames from video."""
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
    """Extract the best single frame from video."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        frame_path = tmp.name

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


def generate_template_thumbnail(
    video_path: str,
    title: str,
    output_path: str,
    style: str = "bold",
) -> str:
    """Generate YouTube thumbnail using local FFmpeg filters."""
    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())

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
            raise RuntimeError(f"Template thumbnail generation failed (code {result.returncode}).")

        return output_path
    finally:
        if os.path.exists(best_frame):
            os.unlink(best_frame)


def generate_ai_thumbnail(
    video_path: str,
    title: str,
    output_path: str,
    context: str = "",
) -> str:
    """Generate a thumbnail using Gemini AI analysis."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY required for AI thumbnail generation")

    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Install google-generativeai: pip install google-generativeai")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    frames = extract_frames(video_path, num_frames=6)
    if not frames:
        raise RuntimeError("Could not extract frames from video")

    try:
        prompt_parts = [
            f"""You are a YouTube thumbnail design expert. Analyze these {len(frames)} frames from a video clip titled "{title}".
{f'Context: {context}' if context else ''}

Pick the BEST frame for a thumbnail (most expressive faces, action, emotion, or visually striking composition).
Suggest a short punchy headline (max 5 words).

Return ONLY a JSON object (no markdown fences):
{{
    "best_frame_index": 0,
    "headline": "YOUR PUNCHY HEADLINE",
    "headline_position": "center",
    "style": "bold",
    "text_color": "#FFFFFF",
    "reasoning": "brief explanation"
}}"""
        ]

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

        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        suggestion = json.loads(text)
        best_idx = max(0, min(suggestion.get("best_frame_index", 0), len(frames) - 1))
        headline = suggestion.get("headline", title[:30])
        position = suggestion.get("headline_position", "center")
        style = suggestion.get("style", "bold")
        text_color = suggestion.get("text_color", "#FFFFFF")

        best_frame_path = frames[best_idx]["path"]
        font_setup = _ffmpeg_font_setup()

        def _esc(txt):
            for ch in ("\\", "'", ":", ";", "[", "]", ","):
                txt = txt.replace(ch, "\\" + ch)
            txt = txt.replace("%", "%%")
            return txt

        safe_headline = _esc(headline.upper())

        if position == "top":
            y_expr = "h*0.15"
        elif position == "bottom":
            y_expr = "h*0.75"
        else:
            y_expr = "(h-text_h)/2"

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
            raise RuntimeError(f"AI thumbnail generation failed (code {result.returncode}).")

        return output_path

    finally:
        for frame in frames:
            if os.path.exists(frame["path"]):
                os.unlink(frame["path"])
