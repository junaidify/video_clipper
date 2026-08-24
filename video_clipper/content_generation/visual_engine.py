"""
Visual Engine Module
Fetches and generates video/image assets for each scene in a script.
Sources:
1. Pexels API (free HD stock video & images)
2. Pixabay API (free HD stock video & images)
3. Text card generator (Pillow + FFmpeg motion animation fallback)
4. Solid color background fallback
"""
import hashlib
import json
import logging
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


@dataclass
class VisualClip:
    """A visual asset prepared for a single scene."""
    scene_number: int
    clip_path: str                  # path to the .mp4 file
    duration: float
    visual_type: str                # 'stock_video', 'stock_image', 'text_card', 'color_bg'
    source: str                     # 'pexels', 'pixabay', 'generated', 'color_bg'
    attribution: str = ""
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "scene_number": self.scene_number,
            "clip_path": self.clip_path,
            "duration": round(self.duration, 1),
            "visual_type": self.visual_type,
            "source": self.source,
            "attribution": self.attribution,
            "success": self.success,
            "error": self.error,
        }


def search_pexels_videos(
    query: str,
    orientation: str = "portrait",
    per_page: int = 3,
) -> List[dict]:
    """Search Pexels API for royalty-free stock videos."""
    api_key = os.environ.get("PEXELS_API_KEY", "")
    if not api_key:
        return []

    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": api_key}
    params = {
        "query": query,
        "orientation": orientation,
        "per_page": per_page,
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            videos = []
            for item in data.get("videos", []):
                files = item.get("video_files", [])
                hd_files = [f for f in files if f.get("quality") == "hd" and f.get("width", 0) >= 720]
                best_file = hd_files[0] if hd_files else (files[0] if files else None)

                if best_file:
                    videos.append({
                        "id": item["id"],
                        "url": best_file["link"],
                        "width": best_file.get("width", 1080),
                        "height": best_file.get("height", 1920),
                        "duration": item.get("duration", 10),
                        "source": "pexels",
                        "user": item.get("user", {}).get("name", "Pexels Creator"),
                    })
            return videos
    except Exception as e:
        logger.warning(f"Pexels video search failed for '{query}': {e}")

    return []


def search_pixabay_videos(
    query: str,
    video_type: str = "all",
    per_page: int = 3,
) -> List[dict]:
    """Search Pixabay API for royalty-free stock videos."""
    api_key = os.environ.get("PIXABAY_API_KEY", "")
    if not api_key:
        return []

    url = "https://pixabay.com/api/videos/"
    params = {
        "key": api_key,
        "q": query,
        "video_type": video_type,
        "per_page": per_page,
        "safesearch": "true",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            videos = []
            for item in data.get("hits", []):
                v_data = item.get("videos", {})
                chosen = v_data.get("large") or v_data.get("medium") or v_data.get("small")
                if chosen and chosen.get("url"):
                    videos.append({
                        "id": item["id"],
                        "url": chosen["url"],
                        "width": chosen.get("width", 1080),
                        "height": chosen.get("height", 1920),
                        "duration": item.get("duration", 10),
                        "source": "pixabay",
                        "user": item.get("user", "Pixabay Creator"),
                    })
            return videos
    except Exception as e:
        logger.warning(f"Pixabay video search failed for '{query}': {e}")

    return []


def download_video_file(url: str, output_path: str) -> bool:
    """Download video file from URL."""
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

        return os.path.isfile(output_path) and os.path.getsize(output_path) > 1000
    except Exception as e:
        logger.error(f"Failed to download video from {url}: {e}")
        return False


def trim_and_format_clip(
    input_path: str,
    duration: float,
    output_path: str,
    width: int = 1080,
    height: int = 1920,
) -> bool:
    """Trim a downloaded stock video to the exact scene duration and format to target aspect ratio."""
    try:
        filter_complex = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,fps=30[vout]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "22", "-an",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=60)
        return result.returncode == 0 and os.path.isfile(output_path)
    except Exception as e:
        logger.error(f"Clip format failed: {e}")
        return False


def image_to_video(
    image_path: str,
    duration: float,
    output_path: str,
    width: int = 1080,
    height: int = 1920,
) -> bool:
    """Convert a static image into an MP4 video clip with subtle motion."""
    try:
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,"
            f"zoompan=z='min(zoom+0.0015,1.15)':d={int(30*duration)}:s={width}x{height}:fps=30"
        )

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", image_path,
            "-vf", vf,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "22", "-pix_fmt", "yuv420p",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=60)
        return result.returncode == 0 and os.path.isfile(output_path)
    except Exception as e:
        logger.error(f"Image to video failed: {e}")
        return False


def generate_color_bg(
    duration: float,
    output_path: str,
    color: str = "#0f172a",
    width: int = 1080,
    height: int = 1920,
) -> bool:
    """Generate a clean solid color background video clip."""
    try:
        color_clean = color.lstrip("#")
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x{color_clean}:s={width}x{height}:r=30",
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "23", "-pix_fmt", "yuv420p",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=30)
        return result.returncode == 0 and os.path.isfile(output_path)
    except Exception as e:
        logger.error(f"Color background generation failed: {e}")
        return False


def generate_text_card(
    headline: str,
    subtext: str,
    duration: float,
    output_path: str,
    width: int = 1080,
    height: int = 1920,
    bg_gradient: tuple = ((15, 23, 42), (30, 41, 59)),
) -> bool:
    """Generate a motion animated text card when stock footage is unavailable."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), bg_gradient[0])
    draw = ImageDraw.Draw(img)

    for y in range(height):
        ratio = y / height
        r = int(bg_gradient[0][0] * (1 - ratio) + bg_gradient[1][0] * ratio)
        g = int(bg_gradient[0][1] * (1 - ratio) + bg_gradient[1][1] * ratio)
        b = int(bg_gradient[0][2] * (1 - ratio) + bg_gradient[1][2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    font_path = None
    for candidate in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if os.path.isfile(candidate):
            font_path = candidate
            break

    try:
        h_font = ImageFont.truetype(font_path, 64) if font_path else ImageFont.load_default()
        s_font = ImageFont.truetype(font_path, 38) if font_path else ImageFont.load_default()
    except Exception:
        h_font = s_font = ImageFont.load_default()

    h_bbox = draw.textbbox((0, 0), headline, font=h_font)
    h_w, h_h = h_bbox[2] - h_bbox[0], h_bbox[3] - h_bbox[1]
    hx = (width - h_w) // 2
    hy = height // 2 - h_h - 20

    draw.text((hx, hy), headline, fill=(255, 255, 255), font=h_font)

    if subtext:
        s_bbox = draw.textbbox((0, 0), subtext, font=s_font)
        s_w = s_bbox[2] - s_bbox[0]
        sx = (width - s_w) // 2
        sy = hy + h_h + 40
        draw.text((sx, sy), subtext, fill=(200, 210, 230), font=s_font)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        card_img_path = tmp.name
    img.save(card_img_path)

    try:
        return image_to_video(card_img_path, duration, output_path, width, height)
    finally:
        if os.path.exists(card_img_path):
            os.unlink(card_img_path)


def fetch_visuals_for_script(
    scenes: List[dict],
    output_dir: str,
    video_format: str = "9:16",
) -> List[VisualClip]:
    """Fetch or generate visual video clips for all scenes in a script."""
    os.makedirs(output_dir, exist_ok=True)

    is_landscape = (video_format == "16:9")
    orientation = "landscape" if is_landscape else "portrait"
    target_w = 1920 if is_landscape else 1080
    target_h = 1080 if is_landscape else 1920

    visual_clips = []

    for scene in scenes:
        scene_num = scene.get("scene_number", 1)
        duration = scene.get("end_time", 5) - scene.get("start_time", 0)
        keywords = scene.get("visual_keywords", [])
        description = scene.get("visual_description", "")
        text_overlay = scene.get("text_overlay", "")
        scene_clip_path = os.path.join(output_dir, f"scene_{scene_num:02d}.mp4")

        logger.info(f"Fetching visuals for Scene {scene_num}: {keywords}")

        video_found = False
        candidates = []

        for kw in keywords[:2]:
            candidates.extend(search_pexels_videos(kw, orientation=orientation))
            if candidates:
                break
            candidates.extend(search_pixabay_videos(kw))
            if candidates:
                break

        if candidates:
            best = candidates[0]
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                raw_stock_path = tmp.name

            if download_video_file(best["url"], raw_stock_path):
                if trim_and_format_clip(raw_stock_path, duration, scene_clip_path, target_w, target_h):
                    video_found = True
                    visual_clips.append(VisualClip(
                        scene_number=scene_num,
                        clip_path=scene_clip_path,
                        duration=duration,
                        visual_type="stock_video",
                        source=best["source"],
                        attribution=f"Video by {best['user']} on {best['source'].capitalize()}",
                        success=True,
                    ))

            if os.path.exists(raw_stock_path):
                os.unlink(raw_stock_path)

        if not video_found:
            headline = text_overlay or (keywords[0] if keywords else f"Scene {scene_num}")
            subtext = description[:50] if description else ""

            if generate_text_card(headline, subtext, duration, scene_clip_path, target_w, target_h):
                visual_clips.append(VisualClip(
                    scene_number=scene_num,
                    clip_path=scene_clip_path,
                    duration=duration,
                    visual_type="text_card",
                    source="generated",
                    success=True,
                ))
            else:
                generate_color_bg(duration, scene_clip_path, width=target_w, height=target_h)
                visual_clips.append(VisualClip(
                    scene_number=scene_num,
                    clip_path=scene_clip_path,
                    duration=duration,
                    visual_type="color_bg",
                    source="color_bg",
                    success=True,
                ))

    return visual_clips
