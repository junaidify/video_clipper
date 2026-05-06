"""
YouTube Downloader Module
Downloads videos from YouTube (and other supported platforms) using yt-dlp.
"""
import json
import logging
import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Regex patterns for supported URL formats
YOUTUBE_PATTERNS = [
    r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+',
    r'(https?://)?(www\.)?youtu\.be/[\w-]+',
    r'(https?://)?(www\.)?youtube\.com/shorts/[\w-]+',
    r'(https?://)?(www\.)?youtube\.com/embed/[\w-]+',
]

SUPPORTED_PATTERNS = YOUTUBE_PATTERNS + [
    r'(https?://)?(www\.)?tiktok\.com/@[\w.-]+/video/\d+',
    r'(https?://)?(www\.)?instagram\.com/(reel|p)/[\w-]+',
    r'(https?://)?(www\.)?twitter\.com/\w+/status/\d+',
    r'(https?://)?(www\.)?x\.com/\w+/status/\d+',
]


@dataclass
class DownloadResult:
    """Result of a video download."""
    success: bool
    file_path: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None


def is_valid_url(url: str) -> bool:
    """Check if URL is a supported video platform link."""
    url = url.strip()
    for pattern in SUPPORTED_PATTERNS:
        if re.match(pattern, url):
            return True
    # Fallback: any URL that looks like a video link
    if re.match(r'https?://', url):
        return True
    return False


def _verify_ytdlp():
    """Ensure yt-dlp is installed."""
    if shutil.which("yt-dlp"):
        return
    # Try importing as Python module
    try:
        import yt_dlp
        return
    except ImportError:
        pass
    raise RuntimeError(
        "yt-dlp is required for URL downloads. Install it:\n"
        "  pip install yt-dlp\n"
        "  or: winget install yt-dlp"
    )


def _has_ffmpeg() -> bool:
    """Check if FFmpeg is installed and on PATH."""
    return shutil.which("ffmpeg") is not None


def get_video_info(url: str) -> dict:
    """Fetch video metadata without downloading."""
    _verify_ytdlp()
    try:
        import yt_dlp
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'description': info.get('description', ''),
                'thumbnail': info.get('thumbnail', ''),
                'url': url,
            }
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {'title': 'Unknown', 'duration': 0, 'url': url, 'error': str(e)}


def download_video(url: str, output_dir: str,
                   max_resolution: int = 1080) -> DownloadResult:
    """
    Download video from URL using yt-dlp.

    Args:
        url: Video URL (YouTube, TikTok, Instagram, etc.)
        output_dir: Directory to save the downloaded video
        max_resolution: Maximum video height (default 1080p)

    Returns:
        DownloadResult with file path and metadata
    """
    _verify_ytdlp()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading video from: {url}")

    try:
        import yt_dlp

        # Output template — sanitized filename
        output_template = str(output_dir / '%(title).80s_%(id)s.%(ext)s')

        ffmpeg_available = _has_ffmpeg()

        if ffmpeg_available:
            # FFmpeg present: download best video+audio separately, merge to MP4
            ydl_opts = {
                'format': f'bestvideo[height<={max_resolution}]+bestaudio/best[height<={max_resolution}]/best',
                'outtmpl': output_template,
                'merge_output_format': 'mp4',
                'quiet': False,
                'no_warnings': False,
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }],
                'geo_bypass': True,
                'max_filesize': 2 * 1024 * 1024 * 1024,
            }
            logger.info("FFmpeg detected — using best quality (separate streams + merge)")
        else:
            # No FFmpeg: download single stream that already has video+audio combined
            # This avoids the merge step entirely
            ydl_opts = {
                'format': f'best[height<={max_resolution}][ext=mp4]/best[height<={max_resolution}]/best',
                'outtmpl': output_template,
                'quiet': False,
                'no_warnings': False,
                'geo_bypass': True,
                'max_filesize': 2 * 1024 * 1024 * 1024,
            }
            logger.warning(
                "FFmpeg NOT found — downloading single combined stream. "
                "Quality may be lower. Install FFmpeg for best results:\n"
                "  Windows: winget install ffmpeg\n"
                "  macOS:   brew install ffmpeg\n"
                "  Linux:   sudo apt install ffmpeg"
            )

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Get the actual downloaded file path
            if info.get('requested_downloads'):
                file_path = info['requested_downloads'][0]['filepath']
            else:
                # Fallback: construct from template
                file_path = ydl.prepare_filename(info)
                # yt-dlp might have changed extension to mp4
                mp4_path = Path(file_path).with_suffix('.mp4')
                if mp4_path.exists():
                    file_path = str(mp4_path)

            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)

            logger.info(f"Download complete: {file_path}")
            logger.info(f"Title: {title}, Duration: {duration}s")

            return DownloadResult(
                success=True,
                file_path=str(file_path),
                title=title,
                duration=duration,
            )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Download failed: {error_msg}")
        return DownloadResult(
            success=False,
            error=error_msg,
        )
