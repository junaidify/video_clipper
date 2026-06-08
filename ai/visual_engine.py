"""
Visual Engine Module
For each scene in a script, pulls stock footage from Pexels/Pixabay
or generates motion graphics (kinetic text, stat cards, transitions).
All rendering via FFmpeg + Pillow.
"""

import json
import logging
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class VisualClip:
    """A single visual clip for one scene."""
    scene_number: int
    clip_path: str              # path to the video/image file
    duration: float             # seconds
    visual_type: str            # 'stock_footage', 'motion_graphic', 'text_animation', 'stat_card'
    source: str                 # 'pexels', 'pixabay', 'generated'
    attribution: str = ""       # credit for stock footage
    width: int = 1080
    height: int = 1920
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "scene_number": self.scene_number,
            "clip_path": self.clip_path,
            "duration": self.duration,
            "visual_type": self.visual_type,
            "source": self.source,
            "attribution": self.attribution,
            "success": self.success,
            "error": self.error,
        }


# ── Stock Footage APIs ──

def search_pexels_videos(keywords: list[str],
                         orientation: str = "portrait",
                         min_duration: int = 5,
                         max_duration: int = 30,
                         per_page: int = 5) -> list[dict]:
    """
    Search Pexels for free stock videos.
    Returns list of {url, width, height, duration, attribution}.
    """
    api_key = os.environ.get("PEXELS_API_KEY", "")
    if not api_key:
        logger.warning("No PEXELS_API_KEY set")
        return []

    query = " ".join(keywords[:4])
    url = "https://api.pexels.com/videos/search"
    params = {
        "query": query,
        "orientation": orientation,
        "per_page": per_page,
        "size": "medium",
    }
    headers = {"Authorization": api_key}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for video in data.get("videos", []):
            dur = video.get("duration", 0)
            if dur < min_duration or dur > max_duration:
                continue

            # Pick the best quality file in portrait/HD
            best_file = None
            for vf in video.get("video_files", []):
                if vf.get("height", 0) >= 720:
                    if best_file is None or vf.get("height", 0) > best_file.get("height", 0):
                        best_file = vf

            if not best_file:
                best_file = video.get("video_files", [{}])[0]

            if best_file and best_file.get("link"):
                results.append({
                    "url": best_file["link"],
                    "width": best_file.get("width", 1080),
                    "height": best_file.get("height", 1920),
                    "duration": dur,
                    "attribution": f"Video by {video.get('user', {}).get('name', 'Unknown')} on Pexels",
                    "source": "pexels",
                })

        logger.info(f"Pexels: {len(results)} videos for '{query}'")
        return results

    except Exception as e:
        logger.error(f"Pexels search failed: {e}")
        return []


def search_pixabay_videos(keywords: list[str],
                          orientation: str = "vertical",
                          min_duration: int = 5,
                          per_page: int = 5) -> list[dict]:
    """
    Search Pixabay for free stock videos.
    """
    api_key = os.environ.get("PIXABAY_API_KEY", "")
    if not api_key:
        logger.warning("No PIXABAY_API_KEY set")
        return []

    query = "+".join(keywords[:4])
    url = "https://pixabay.com/api/videos/"
    params = {
        "key": api_key,
        "q": query,
        "per_page": per_page,
        "safesearch": "true",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for hit in data.get("hits", []):
            dur = hit.get("duration", 0)
            if dur < min_duration:
                continue

            # Get medium quality video
            videos = hit.get("videos", {})
            medium = videos.get("medium", {}) or videos.get("small", {})

            if medium and medium.get("url"):
                results.append({
                    "url": medium["url"],
                    "width": medium.get("width", 1080),
                    "height": medium.get("height", 1920),
                    "duration": dur,
                    "attribution": f"Video by {hit.get('user', 'Unknown')} on Pixabay",
                    "source": "pixabay",
                })

        logger.info(f"Pixabay: {len(results)} videos for '{query}'")
        return results

    except Exception as e:
        logger.error(f"Pixabay search failed: {e}")
        return []


def download_stock_video(video_url: str, output_path: str, timeout: int = 60) -> bool:
    """Download a stock video file."""
    try:
        resp = requests.get(video_url, stream=True, timeout=timeout)
        resp.raise_for_status()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return os.path.isfile(output_path) and os.path.getsize(output_path) > 1000

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


# ── Motion Graphics Generation ──
# Windows-safe approach: Pillow renders text to PNG, FFmpeg animates it.
# This avoids all drawtext escaping and fontconfig issues on Windows.


def _find_system_font() -> str:
    """Find a usable system font file on Windows/Linux/Mac."""
    candidates = [
        # Windows
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\verdana.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # Mac
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
    ]
    for f in candidates:
        if os.path.isfile(f):
            return f
    return ""


def _render_text_frame(text: str, width: int, height: int,
                       font_size: int, fg_color: str, bg_color: str,
                       output_png: str) -> bool:
    """Render text onto a PNG image using Pillow. Handles all escaping safely."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Parse colors
        bg = _parse_color(bg_color)
        fg = _parse_color(fg_color)

        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = _find_system_font()
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Word wrap
        chars_per_line = max(10, int(width / (font_size * 0.55)))
        wrapped = _word_wrap(text, chars_per_line)

        # Measure text
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Center text
        x = (width - tw) // 2
        y = (height - th) // 2

        # Draw text with outline for readability
        outline_color = (0, 0, 0) if fg != (0, 0, 0) else (255, 255, 255)
        for ox, oy in [(-2, -2), (-2, 2), (2, -2), (2, 2), (-2, 0), (2, 0), (0, -2), (0, 2)]:
            draw.multiline_text((x + ox, y + oy), wrapped, font=font,
                                fill=outline_color, align="center")
        draw.multiline_text((x, y), wrapped, font=font, fill=fg, align="center")

        img.save(output_png, "PNG")
        return True

    except ImportError:
        logger.warning("Pillow not installed, falling back to FFmpeg-only")
        return False
    except Exception as e:
        logger.error(f"Text render error: {e}")
        return False


def _parse_color(color_str: str) -> tuple:
    """Parse color string to RGB tuple."""
    color_str = color_str.strip().lower()
    named = {
        "white": (255, 255, 255), "black": (0, 0, 0),
        "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
        "yellow": (255, 255, 0), "cyan": (0, 255, 255),
    }
    if color_str in named:
        return named[color_str]
    if color_str.startswith("#") and len(color_str) >= 7:
        return (int(color_str[1:3], 16), int(color_str[3:5], 16), int(color_str[5:7], 16))
    return (255, 255, 255)


def generate_text_animation(text: str, duration: float, output_path: str,
                            width: int = 1080, height: int = 1920,
                            font_size: int = 72, color: str = "white",
                            bg_color: str = "black",
                            animation: str = "fade_in") -> bool:
    """
    Generate a text animation video.
    Strategy: Pillow renders text to PNG → FFmpeg animates (fade/zoom/loop).
    Windows-safe — no drawtext filter or fontconfig needed.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Clean text for display
    clean_text = text.replace("'", "'").replace('"', '"').strip()
    if not clean_text:
        clean_text = "..."

    tmp_dir = tempfile.mkdtemp()
    text_png = os.path.join(tmp_dir, "text_frame.png")

    try:
        # Step 1: Render text to PNG via Pillow
        rendered = _render_text_frame(clean_text, width, height,
                                      font_size, color, bg_color, text_png)

        if not rendered or not os.path.isfile(text_png):
            # Pillow failed — use pure FFmpeg color background (no text)
            logger.warning("Pillow render failed, generating plain background")
            return generate_color_bg(duration, output_path, bg_color, width, height)

        # Step 2: Animate the PNG with FFmpeg
        if animation == "fade_in":
            vf = f"fade=t=in:st=0:d=1,fade=t=out:st={max(0, duration-0.5)}:d=0.5"
        elif animation == "slide_up":
            # Zoom in slightly for motion effect
            vf = (
                f"zoompan=z='min(1.08,1+0.08*on/({int(30*duration)}))':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={int(30*duration)}:s={width}x{height}:fps=30,"
                f"fade=t=in:st=0:d=0.5"
            )
        else:
            vf = f"fade=t=in:st=0:d=0.8"

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", text_png,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=60)

        if result.returncode != 0:
            logger.error(f"Text animation FFmpeg failed: {result.stderr[-300:]}")
            return False

        return os.path.isfile(output_path) and os.path.getsize(output_path) > 100

    except Exception as e:
        logger.error(f"Text animation error: {e}")
        return False
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def generate_stat_card(stat_text: str, label: str, duration: float,
                       output_path: str, width: int = 1080,
                       height: int = 1920) -> bool:
    """
    Generate a stat card video — large stat with label below.
    Uses Pillow for rendering, FFmpeg for animation.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    tmp_dir = tempfile.mkdtemp()
    card_png = os.path.join(tmp_dir, "stat_card.png")

    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (width, height), (26, 26, 46))  # #1a1a2e
        draw = ImageDraw.Draw(img)

        font_path = _find_system_font()
        try:
            stat_font = ImageFont.truetype(font_path, 96) if font_path else ImageFont.load_default()
            label_font = ImageFont.truetype(font_path, 42) if font_path else ImageFont.load_default()
        except Exception:
            stat_font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        # Draw stat text (cyan)
        stat_bbox = draw.textbbox((0, 0), stat_text, font=stat_font)
        sw = stat_bbox[2] - stat_bbox[0]
        sx = (width - sw) // 2
        sy = height // 2 - 80
        draw.text((sx, sy), stat_text, font=stat_font, fill=(0, 212, 255))

        # Draw label (gray)
        label_bbox = draw.textbbox((0, 0), label, font=label_font)
        lw = label_bbox[2] - label_bbox[0]
        lx = (width - lw) // 2
        ly = height // 2 + 30
        draw.text((lx, ly), label, font=label_font, fill=(204, 204, 204))

        img.save(card_png, "PNG")

        # Animate with fade in
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", card_png,
            "-vf", f"fade=t=in:st=0:d=0.5,fade=t=out:st={max(0, duration-0.5)}:d=0.5",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=60)
        return result.returncode == 0

    except ImportError:
        logger.warning("Pillow not available for stat card, using plain bg")
        return generate_color_bg(duration, output_path)
    except Exception as e:
        logger.error(f"Stat card error: {e}")
        return False
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def generate_color_bg(duration: float, output_path: str,
                      color: str = "#1a1a2e",
                      width: int = 1080, height: int = 1920) -> bool:
    """Generate a plain colored background video as fallback."""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s={width}x{height}:d={duration}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=30)
        return result.returncode == 0
    except Exception:
        return False


# ── Format Conversion ──

def convert_to_portrait(input_path: str, output_path: str,
                        width: int = 1080, height: int = 1920) -> bool:
    """
    Convert any video to 9:16 portrait format.
    Uses crop/pad to fit without distortion.
    """
    try:
        # Probe input dimensions
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "v", input_path],
            capture_output=True, encoding='utf-8', errors='replace'
        )
        info = json.loads(probe.stdout)
        streams = info.get("streams", [])
        if not streams:
            return False

        in_w = streams[0].get("width", 1920)
        in_h = streams[0].get("height", 1080)

        # Calculate scaling: fit to portrait, then crop excess
        target_ratio = width / height  # 0.5625 for 9:16
        input_ratio = in_w / in_h

        if input_ratio > target_ratio:
            # Input is wider — scale by height, crop width
            scale_filter = f"scale=-2:{height}"
        else:
            # Input is taller — scale by width, crop height
            scale_filter = f"scale={width}:-2"

        vf = f"{scale_filter},crop={width}:{height},setsar=1"

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-an",  # strip audio — we'll add TTS later
            "-pix_fmt", "yuv420p",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=120)
        return result.returncode == 0

    except Exception as e:
        logger.error(f"Portrait conversion failed: {e}")
        return False


def trim_clip(input_path: str, output_path: str,
              duration: float) -> bool:
    """Trim a clip to exact duration."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-an", "-pix_fmt", "yuv420p",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=60)
        return result.returncode == 0
    except Exception:
        return False


# ── Master Visual Pipeline ──

def fetch_visuals_for_script(scenes: list[dict],
                             output_dir: str) -> list[VisualClip]:
    """
    For each scene in a script, fetch/generate the appropriate visual.
    AI decides stock footage vs motion graphic per scene.

    Args:
        scenes: List of scene dicts (from Script.to_dict()['scenes'])
        output_dir: Directory to save visual clips

    Returns:
        List of VisualClip objects, one per scene
    """
    os.makedirs(output_dir, exist_ok=True)
    clips = []

    for scene in scenes:
        sn = scene["scene_number"]
        duration = scene.get("duration", scene["end_time"] - scene["start_time"])
        visual_type = scene.get("visual_type", "stock_footage")
        keywords = scene.get("visual_keywords", [])
        description = scene.get("visual_description", "")
        text_overlay = scene.get("text_overlay", "")
        clip_path = os.path.join(output_dir, f"scene_{sn:02d}.mp4")

        logger.info(f"Scene {sn}: fetching {visual_type} ({duration:.1f}s)")

        if visual_type == "stock_footage":
            clip = _fetch_stock_for_scene(sn, keywords, duration, clip_path)
        elif visual_type == "text_animation":
            text = text_overlay or description[:60]
            success = generate_text_animation(
                text, duration, clip_path,
                animation="fade_in"
            )
            clip = VisualClip(
                scene_number=sn, clip_path=clip_path,
                duration=duration, visual_type="text_animation",
                source="generated", success=success,
                error="" if success else "Text animation generation failed"
            )
        elif visual_type == "stat_card":
            stat = text_overlay or description[:30]
            label = " ".join(keywords[:3]) if keywords else "Trending"
            success = generate_stat_card(stat, label, duration, clip_path)
            clip = VisualClip(
                scene_number=sn, clip_path=clip_path,
                duration=duration, visual_type="stat_card",
                source="generated", success=success,
                error="" if success else "Stat card generation failed"
            )
        elif visual_type == "motion_graphic":
            # For motion graphics, generate text animation with the description
            text = text_overlay or description[:80]
            success = generate_text_animation(
                text, duration, clip_path,
                animation="slide_up", font_size=60
            )
            clip = VisualClip(
                scene_number=sn, clip_path=clip_path,
                duration=duration, visual_type="motion_graphic",
                source="generated", success=success,
                error="" if success else "Motion graphic generation failed"
            )
        else:
            # Unknown type — generate placeholder
            success = generate_color_bg(duration, clip_path)
            clip = VisualClip(
                scene_number=sn, clip_path=clip_path,
                duration=duration, visual_type="placeholder",
                source="generated", success=success,
            )

        # Fallback: if stock fetch failed, generate a text card
        if not clip.success:
            logger.warning(f"Scene {sn}: primary visual failed, using fallback")
            fallback_text = text_overlay or description[:60] or "..."
            success = generate_text_animation(
                fallback_text, duration, clip_path,
                animation="fade_in", font_size=54
            )
            clip = VisualClip(
                scene_number=sn, clip_path=clip_path,
                duration=duration, visual_type="text_fallback",
                source="generated", success=success,
                error="" if success else "All visual methods failed"
            )

        clips.append(clip)

    success_count = sum(1 for c in clips if c.success)
    logger.info(f"Visual engine: {success_count}/{len(clips)} scenes ready")
    return clips


def _fetch_stock_for_scene(scene_number: int, keywords: list[str],
                           duration: float, output_path: str) -> VisualClip:
    """Try Pexels first, then Pixabay, then generate fallback."""

    # Try Pexels
    results = search_pexels_videos(keywords, min_duration=int(duration))
    if not results:
        # Try Pixabay
        results = search_pixabay_videos(keywords, min_duration=int(duration))

    if not results:
        # Try with fewer/broader keywords
        broad = keywords[:1] if keywords else ["abstract background"]
        results = search_pexels_videos(broad, min_duration=3)
        if not results:
            results = search_pixabay_videos(broad, min_duration=3)

    if results:
        # Download the best match
        best = results[0]
        raw_path = output_path.replace(".mp4", "_raw.mp4")

        if download_stock_video(best["url"], raw_path):
            # Convert to portrait + trim to exact duration
            converted = convert_to_portrait(raw_path, output_path)
            if converted:
                trim_clip(output_path, output_path + ".tmp", duration)
                if os.path.isfile(output_path + ".tmp"):
                    os.replace(output_path + ".tmp", output_path)

            # Clean up raw
            if os.path.isfile(raw_path):
                os.remove(raw_path)

            if os.path.isfile(output_path):
                return VisualClip(
                    scene_number=scene_number,
                    clip_path=output_path,
                    duration=duration,
                    visual_type="stock_footage",
                    source=best.get("source", "pexels"),
                    attribution=best.get("attribution", ""),
                    success=True,
                )

    # All stock sources failed
    return VisualClip(
        scene_number=scene_number,
        clip_path=output_path,
        duration=duration,
        visual_type="stock_footage",
        source="none",
        success=False,
        error="No stock footage found for keywords",
    )


# ── Helpers ──

def _word_wrap(text: str, max_chars: int) -> str:
    """Wrap text at word boundaries."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "\n".join(lines)
