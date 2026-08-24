"""
Universal Video Downloader Module
Powered by yt-dlp. Downloads video/audio from 1,800+ supported sites:
YouTube, TikTok, Instagram, Twitter/X, Facebook, Reddit, Vimeo, Twitch, Rumble, Bilibili, and more.
Includes robust browser cookie fallback, client hints, and error diagnostics.
"""
import glob
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Cache of browser names that failed cookie decryption
_browser_blacklist = set()

# Platforms requiring special user guidance or cookie auth
PLATFORM_HINTS = {
    "instagram": {
        "name": "Instagram",
        "cookie_needed": True,
        "tip": "Instagram often requires login. If it fails, log into Instagram in Chrome/Edge/Firefox on this computer.",
    },
    "facebook": {
        "name": "Facebook",
        "cookie_needed": True,
        "tip": "Private/group Facebook videos require login cookies.",
    },
    "twitter": {
        "name": "X / Twitter",
        "cookie_needed": False,
        "tip": "Direct tweet links with video work best.",
    },
    "tiktok": {
        "name": "TikTok",
        "cookie_needed": False,
        "tip": "Standard TikTok video URLs work directly.",
    },
    "youtube": {
        "name": "YouTube",
        "cookie_needed": False,
        "tip": "Standard YouTube, Shorts, and Live replay links work.",
    },
    "reddit": {
        "name": "Reddit",
        "cookie_needed": False,
        "tip": "Post URLs containing v.redd.it videos work directly.",
    },
    "twitch": {
        "name": "Twitch",
        "cookie_needed": False,
        "tip": "Twitch clips and VODs are supported.",
    },
    "vimeo": {
        "name": "Vimeo",
        "cookie_needed": False,
        "tip": "Public Vimeo videos work directly.",
    },
}

DRM_PLATFORMS = {
    "netflix.com": "Netflix",
    "disneyplus.com": "Disney+",
    "hulu.com": "Hulu",
    "hbomax.com": "Max / HBO",
    "max.com": "Max",
    "primevideo.com": "Amazon Prime Video",
    "apple.com/apple-tv-plus": "Apple TV+",
    "peacocktv.com": "Peacock",
    "paramountplus.com": "Paramount+",
    "hotstar.com": "JioHotstar",
    "jiocinema.com": "JioCinema",
    "zee5.com": "ZEE5",
    "sonyliv.com": "SonyLIV",
}


@dataclass
class DownloadResult:
    """Result of a video download operation."""
    success: bool
    file_path: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[float] = None
    uploader: Optional[str] = None
    channel_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    platform: Optional[str] = None
    file_size_mb: float = 0.0
    error: Optional[str] = None
    cookie_used: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "file_path": self.file_path,
            "title": self.title,
            "duration": self.duration,
            "uploader": self.uploader,
            "channel_url": self.channel_url,
            "thumbnail_url": self.thumbnail_url,
            "platform": self.platform,
            "file_size_mb": self.file_size_mb,
            "error": self.error,
            "cookie_used": self.cookie_used,
        }


def is_valid_url(url: str) -> bool:
    """Validate URL syntax."""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def is_drm_platform(url: str) -> Optional[str]:
    """Check if the URL belongs to a known DRM-protected streaming service."""
    url_lower = url.lower()

    if "amazon." in url_lower and ("/video/" in url_lower or "/dp/" in url_lower or "/gp/video" in url_lower):
        return "Amazon Prime Video"

    for domain, name in DRM_PLATFORMS.items():
        if domain in url_lower:
            return name
    return None


def _extract_domain(url: str) -> str:
    """Extract base domain from URL (e.g. 'youtube.com')."""
    if not url:
        return ""
    try:
        if "://" not in url:
            url = "https://" + url
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        if netloc.startswith("m."):
            netloc = netloc[2:]
        return netloc
    except Exception:
        return ""


def _detect_platform(url: str) -> str:
    """Detect human-readable platform name from URL."""
    domain = _extract_domain(url)
    mappings = {
        "youtube.com": "YouTube",
        "youtu.be": "YouTube",
        "tiktok.com": "TikTok",
        "instagram.com": "Instagram",
        "twitter.com": "X/Twitter",
        "x.com": "X/Twitter",
        "facebook.com": "Facebook",
        "fb.watch": "Facebook",
        "reddit.com": "Reddit",
        "v.redd.it": "Reddit",
        "vimeo.com": "Vimeo",
        "twitch.tv": "Twitch",
        "rumble.com": "Rumble",
        "bilibili.com": "Bilibili",
        "dailymotion.com": "Dailymotion",
        "pinterest.com": "Pinterest",
        "pin.it": "Pinterest",
        "linkedin.com": "LinkedIn",
        "threads.net": "Threads",
        "bluesky.app": "Bluesky",
        "bsky.app": "Bluesky",
    }
    for d, name in mappings.items():
        if d in domain:
            return name
    return domain.capitalize() if domain else "Unknown"


def _get_cookie_sources() -> list:
    """
    Return list of cookie sources to try, in priority order:
    1. Explicit cookies.txt file in project root
    2. Explicit cookie file from env var YOUTUBE_COOKIES_FILE
    3. Browser cookie extraction (chrome, edge, firefox, etc.)
    """
    sources = []

    env_cookie = os.environ.get("YOUTUBE_COOKIES_FILE", "")
    if env_cookie and os.path.isfile(env_cookie):
        sources.append(("file", env_cookie))

    project_root = Path(__file__).resolve().parent.parent.parent
    for fname in ["cookies.txt", "youtube_cookies.txt", "youtube.com_cookies.txt"]:
        cpath = project_root / fname
        if cpath.is_file():
            sources.append(("file", str(cpath)))

    cur_cpath = Path.cwd() / "cookies.txt"
    if cur_cpath.is_file() and ("file", str(cur_cpath)) not in sources:
        sources.append(("file", str(cur_cpath)))

    # Installed desktop browsers
    for browser in ["chrome", "edge", "firefox", "brave", "opera", "chromium", "vivaldi"]:
        if browser not in _browser_blacklist:
            sources.append(("browser", browser))

    return sources


def _build_ydl_opts(
    output_template: str,
    cookie_source: Optional[Tuple[str, str]] = None,
    progress_hook: Optional[Callable] = None,
    format_spec: str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
) -> dict:
    """Build yt-dlp options dictionary."""
    opts = {
        "outtmpl": output_template,
        "format": format_spec,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
        "overwrites": True,
        "geo_bypass": True,
        "nocheckcertificate": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "mweb", "web_embedded", "tv", "web"],
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
        "postprocessors": [{
            "key": "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }],
    }

    if progress_hook:
        opts["progress_hooks"] = [progress_hook]

    if cookie_source:
        stype, sval = cookie_source
        if stype == "file":
            opts["cookiefile"] = sval
            logger.info(f"Using cookie file: {sval}")
        elif stype == "browser":
            opts["cookiesfrombrowser"] = (sval, None, None, None)
            logger.info(f"Extracting cookies from browser: {sval}")

    return opts


def _is_bot_detection_error(error_msg: str) -> bool:
    """Check if the error is caused by platform bot/login detection."""
    msg = error_msg.lower()
    signals = [
        "sign in to confirm you're not a bot",
        "sign in to confirm your age",
        "login required",
        "this video is private",
        "requires authentication",
        "http error 429",
        "too many requests",
        "bot detection",
        "consent",
        "confirm you're not a bot",
        "unable to extract",
        "challenge",
    ]
    return any(s in msg for s in signals)


def _is_dpapi_error(error_msg: str) -> bool:
    """Check if the error was DPAPI/cookie decryption failure on Windows."""
    msg = error_msg.lower()
    return "dpapi" in msg or "cryptunprotectdata" in msg or "could not decrypt" in msg


def get_video_info(url: str) -> Optional[dict]:
    """Fetch video metadata without downloading the full media file."""
    if not is_valid_url(url):
        return None

    drm = is_drm_platform(url)
    if drm:
        return {
            "title": f"[{drm}] Protected Content",
            "duration": 0,
            "uploader": drm,
            "platform": drm,
            "thumbnail": "",
            "is_drm": True,
            "error": f"{drm} uses DRM protection. Streaming services cannot be downloaded.",
        }

    import yt_dlp

    cookie_sources = [None] + _get_cookie_sources()

    for cookie_src in cookie_sources:
        try:
            opts = {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "socket_timeout": 15,
                "extract_flat": "in_playlist",
                "geo_bypass": True,
                "nocheckcertificate": True,
                "extractor_args": {
                    "youtube": {
                        "player_client": ["android", "ios", "mweb", "web_embedded", "tv", "web"],
                    }
                },
                "http_headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            }
            if cookie_src:
                stype, sval = cookie_src
                if stype == "file":
                    opts["cookiefile"] = sval
                elif stype == "browser":
                    opts["cookiesfrombrowser"] = (sval, None, None, None)

            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    continue

                return {
                    "title": info.get("title", "Untitled Video"),
                    "duration": float(info.get("duration") or 0),
                    "uploader": info.get("uploader") or info.get("channel") or info.get("uploader_id", "Unknown"),
                    "channel_url": info.get("channel_url") or info.get("uploader_url", ""),
                    "thumbnail": info.get("thumbnail", ""),
                    "view_count": info.get("view_count"),
                    "platform": _detect_platform(url),
                    "is_drm": False,
                    "error": None,
                }
        except Exception as e:
            err_str = str(e)
            if cookie_src and cookie_src[0] == "browser" and _is_dpapi_error(err_str):
                _browser_blacklist.add(cookie_src[1])
            logger.debug(f"get_video_info attempt failed: {err_str[:150]}")
            continue

    return {
        "title": "Unknown Video",
        "duration": 0,
        "uploader": "Unknown",
        "platform": _detect_platform(url),
        "thumbnail": "",
        "is_drm": False,
        "error": "Could not fetch video info. The video may be private or unavailable.",
    }


def download_video(
    url: str,
    output_dir: str,
    custom_filename: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> DownloadResult:
    """
    Download a video from any supported URL to the local disk.

    Features:
    - Multi-client fallback (iOS, Android, Web)
    - Browser cookie auto-discovery
    - Real-time progress callback

    Args:
        url: Web URL of the video.
        output_dir: Destination directory.
        custom_filename: Optional base filename (without extension).
        progress_callback: Optional fn(percentage: float, status_msg: str).

    Returns:
        DownloadResult
    """
    url = url.strip()
    platform = _detect_platform(url)

    if not is_valid_url(url):
        return DownloadResult(
            success=False, platform=platform,
            error=f"Invalid URL: '{url}'. Must start with http:// or https://",
        )

    drm_name = is_drm_platform(url)
    if drm_name:
        return DownloadResult(
            success=False, platform=drm_name,
            error=f"{drm_name} uses DRM encryption (Widevine/FairPlay). "
                  f"Subscription streaming platforms cannot be downloaded.",
        )

    os.makedirs(output_dir, exist_ok=True)

    if custom_filename:
        safe_name = re.sub(r'[^\w\-]', '_', custom_filename)
        output_template = os.path.join(output_dir, f"{safe_name}.%(ext)s")
    else:
        output_template = os.path.join(output_dir, "%(title).60s_%(id)s.%(ext)s")

    last_pct = [0.0]

    def _progress_hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                pct = round((downloaded / total) * 100, 1)
                if pct != last_pct[0]:
                    last_pct[0] = pct
                    speed = d.get("speed", 0) or 0
                    speed_mb = speed / (1024 * 1024) if speed else 0
                    eta = d.get("eta", 0) or 0
                    msg = f"Downloading: {pct:.0f}% ({speed_mb:.1f} MB/s, ETA {eta}s)"
                    if progress_callback:
                        progress_callback(pct, msg)
        elif d["status"] == "finished":
            if progress_callback:
                progress_callback(95.0, "Processing and converting video container...")

    import yt_dlp

    attempts = [
        (None, "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best", "Standard"),
        (None, "best/bestvideo+bestaudio", "Fallback format"),
    ]

    for csource in _get_cookie_sources():
        label = f"Cookie ({csource[0]}:{csource[1]})"
        attempts.append((csource, "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best", label))

    last_error = ""
    for cookie_src, fmt, attempt_label in attempts:
        if cookie_src and cookie_src[0] == "browser" and cookie_src[1] in _browser_blacklist:
            continue

        logger.info(f"Download attempt [{attempt_label}] for {url}")
        if progress_callback:
            progress_callback(5.0, f"Connecting to {platform} ({attempt_label})...")

        ydl_opts = _build_ydl_opts(
            output_template,
            cookie_source=cookie_src,
            progress_hook=_progress_hook,
            format_spec=fmt,
        )

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    last_error = "yt-dlp returned empty info"
                    continue

                title = info.get("title", "Untitled")
                duration = float(info.get("duration") or 0)
                uploader = info.get("uploader") or info.get("channel") or "Unknown"
                channel_url = info.get("channel_url") or info.get("uploader_url") or ""
                thumbnail_url = info.get("thumbnail") or ""

                expected_path = ydl.prepare_filename(info)
                mp4_path = os.path.splitext(expected_path)[0] + ".mp4"

                actual_path = None
                for candidate in [mp4_path, expected_path]:
                    if os.path.isfile(candidate):
                        actual_path = candidate
                        break

                if not actual_path:
                    stem = os.path.splitext(os.path.basename(expected_path))[0]
                    matches = glob.glob(os.path.join(output_dir, f"{glob.escape(stem)}*"))
                    video_matches = [m for m in matches if m.lower().endswith(('.mp4', '.mkv', '.webm', '.mov', '.avi'))]
                    if video_matches:
                        actual_path = video_matches[0]

                if not actual_path or not os.path.isfile(actual_path):
                    last_error = "Downloaded file could not be located on disk."
                    continue

                if not actual_path.endswith(".mp4"):
                    converted_path = os.path.splitext(actual_path)[0] + "_converted.mp4"
                    cmd = ["ffmpeg", "-y", "-i", actual_path, "-c", "copy", "-movflags", "+faststart", converted_path]
                    subprocess.run(cmd, capture_output=True, timeout=120)
                    if os.path.isfile(converted_path):
                        try:
                            os.unlink(actual_path)
                        except OSError:
                            pass
                        actual_path = converted_path

                file_size_mb = round(os.path.getsize(actual_path) / (1024 * 1024), 2)
                cookie_label = f"{cookie_src[0]}:{cookie_src[1]}" if cookie_src else None

                logger.info(f"Download successful: '{title}' ({file_size_mb:.1f} MB) -> {actual_path}")
                if progress_callback:
                    progress_callback(100.0, f"Download complete: {title[:40]} ({file_size_mb:.1f} MB)")

                return DownloadResult(
                    success=True,
                    file_path=actual_path,
                    title=title,
                    duration=duration,
                    uploader=uploader,
                    channel_url=channel_url,
                    thumbnail_url=thumbnail_url,
                    platform=platform,
                    file_size_mb=file_size_mb,
                    cookie_used=cookie_label,
                )

        except Exception as e:
            err_msg = str(e)
            last_error = err_msg

            if cookie_src and cookie_src[0] == "browser" and _is_dpapi_error(err_msg):
                _browser_blacklist.add(cookie_src[1])
                logger.warning(f"Blacklisting browser cookies for '{cookie_src[1]}' due to DPAPI decryption failure.")

            logger.warning(f"Download attempt failed [{attempt_label}]: {err_msg[:200]}")
            continue

    hint = PLATFORM_HINTS.get(platform.lower(), {})
    user_tip = hint.get("tip", "")

    clean_err = _clean_error_message(last_error, platform, user_tip)

    logger.error(f"All download attempts failed for {url}: {clean_err}")
    if progress_callback:
        progress_callback(0.0, f"Download failed: {clean_err[:60]}")

    return DownloadResult(
        success=False,
        platform=platform,
        error=clean_err,
    )


def _clean_error_message(raw_error: str, platform: str, tip: str = "") -> str:
    """Format raw yt-dlp error into user-actionable message."""
    raw_lower = raw_error.lower()

    if "sign in to confirm you're not a bot" in raw_lower:
        base = f"{platform} blocked this request with bot detection."
        solution = " Solution: Export your YouTube cookies using a browser extension (e.g. 'Get cookies.txt LOCALLY') and place cookies.txt in the project directory."
        return base + solution

    if "video unavailable" in raw_lower or "this video is unavailable" in raw_lower:
        return f"This video is unavailable on {platform}. It may have been deleted, set to private, or region-restricted."

    if "private video" in raw_lower:
        return f"This is a private video on {platform}. You must provide cookies from an authorized logged-in account to download it."

    if "login required" in raw_lower:
        return f"{platform} requires login to access this video. {tip or 'Please provide cookies.txt.'}"

    if "http error 429" in raw_lower or "too many requests" in raw_lower:
        return f"{platform} rate-limited requests (HTTP 429). Please wait a few minutes before trying again."

    clean = re.sub(r'ERROR:\s*\[.*?\]\s*', '', raw_error)
    clean = re.sub(r'\s*\(caused by.*?\)', '', clean)
    clean = clean.strip()

    if tip:
        clean = f"{clean}. Tip: {tip}"

    return clean or f"Download failed from {platform}."
