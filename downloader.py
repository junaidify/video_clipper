"""
Universal Video Downloader Module
Downloads videos from 1800+ platforms using yt-dlp.

Supported platforms include (but are not limited to):
  YouTube, TikTok, Instagram, X/Twitter, Facebook, Vimeo, Dailymotion,
  Reddit, Twitch, Bilibili, SoundCloud, LinkedIn (public), Rumble,
  Kick, Odysee, PeerTube, VK, Naver, Weibo, and many more.

NOT supported (DRM-protected):
  Netflix, Disney+, Amazon Prime, Hulu, HBO Max, Apple TV+, Paramount+

See full list: https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md
"""
import json
import logging
import os
import re
import subprocess
import shutil
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ─── Platform-specific hints for better error messages ───

PLATFORM_HINTS = {
    'instagram.com': {
        'name': 'Instagram',
        'login_tip': 'Private content requires cookies. Export cookies from your browser and set COOKIES_FILE in .env.',
        'known_issues': 'Instagram aggressively blocks scrapers. If downloads fail, try again later or use cookies.',
    },
    'tiktok.com': {
        'name': 'TikTok',
        'login_tip': 'Most TikTok videos are public and download fine without login.',
        'known_issues': 'TikTok rotates their API frequently. Keep yt-dlp updated: pip install -U yt-dlp',
    },
    'twitter.com': {
        'name': 'X/Twitter',
        'login_tip': 'Some tweets require login. Export cookies from your browser.',
        'known_issues': 'X rate-limits aggressively. Cookies from a logged-in session help.',
    },
    'x.com': {
        'name': 'X/Twitter',
        'login_tip': 'Some posts require login. Export cookies from your browser.',
        'known_issues': 'X rate-limits aggressively. Cookies from a logged-in session help.',
    },
    'facebook.com': {
        'name': 'Facebook',
        'login_tip': 'Private/friends-only videos require cookies from a logged-in session.',
        'known_issues': 'Only public videos work without authentication.',
    },
    'linkedin.com': {
        'name': 'LinkedIn',
        'login_tip': 'LinkedIn videos almost always require cookies from a logged-in session.',
        'known_issues': 'LinkedIn blocks most scraping. Cookie auth is usually required.',
    },
    'twitch.tv': {
        'name': 'Twitch',
        'login_tip': 'VODs and clips are usually public. Sub-only VODs need cookies.',
        'known_issues': 'Live streams can be downloaded but will run until the stream ends.',
    },
    'bilibili.com': {
        'name': 'Bilibili',
        'login_tip': 'Some videos require login for higher quality. Export cookies if needed.',
        'known_issues': 'Geo-restricted content may fail without a Chinese IP.',
    },
}

# YouTube player clients to try (in order) when bot detection kicks in.
# 'mediaconnect' bypasses most bot detection (including Made-for-Kids COPPA restrictions).
# 'android' uses the mobile API which has looser restrictions.
# 'web' is the default but triggers bot detection on datacenter IPs and kids content.
_YT_PLAYER_CLIENTS = ['mediaconnect', 'android', 'web']

# Platforms that use DRM — will never work
# NOTE: use domain names only (not URL paths) — _extract_domain returns hostname
DRM_DOMAINS = {
    'netflix.com', 'disneyplus.com', 'disney.com', 'hulu.com',
    'hbomax.com', 'max.com', 'primevideo.com', 'amazon.com',
    'tv.apple.com', 'paramountplus.com', 'peacocktv.com',
    'crunchyroll.com', 'funimation.com', 'discoveryplus.com',
}

# Amazon URLs that are NOT DRM (e.g. product pages, shopping)
# Only amazon.com/gp/video and amazon.com/*/dp/ (Prime Video) are DRM
_AMAZON_VIDEO_PATTERNS = ['/gp/video', '/dp/', '/watch/', '/video/']


@dataclass
class DownloadResult:
    """Result of a video download."""
    success: bool
    file_path: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    platform: Optional[str] = None
    warning: Optional[str] = None


def _extract_domain(url: str) -> str:
    """Extract the base domain from a URL."""
    try:
        parsed = urlparse(url if '://' in url else f'https://{url}')
        host = parsed.hostname or ''
        # Strip 'www.' and 'm.' prefixes
        for prefix in ('www.', 'm.', 'mobile.'):
            if host.startswith(prefix):
                host = host[len(prefix):]
        return host.lower()
    except Exception:
        return ''


def _detect_platform(url: str) -> str:
    """Detect platform name from URL for logging/UI."""
    domain = _extract_domain(url)
    for key, info in PLATFORM_HINTS.items():
        if key in domain:
            return info['name']
    # Common platforms without special hints
    platform_map = {
        'youtube.com': 'YouTube', 'youtu.be': 'YouTube',
        'vimeo.com': 'Vimeo', 'dailymotion.com': 'Dailymotion',
        'reddit.com': 'Reddit', 'rumble.com': 'Rumble',
        'kick.com': 'Kick', 'odysee.com': 'Odysee',
        'soundcloud.com': 'SoundCloud', 'bandcamp.com': 'Bandcamp',
        'vk.com': 'VK', 'weibo.com': 'Weibo',
        'naver.com': 'Naver', 'nicovideo.jp': 'Niconico',
        'archive.org': 'Internet Archive',
    }
    for key, name in platform_map.items():
        if key in domain:
            return name
    return domain.split('.')[0].capitalize() if domain else 'Unknown'


def is_valid_url(url: str) -> bool:
    """
    Check if URL is a potentially valid video link.
    Accepts any HTTP/HTTPS URL — yt-dlp will handle the rest.
    Rejects known DRM platforms early with a clear message.
    """
    url = url.strip()
    if not re.match(r'https?://', url):
        return False
    return True


def is_drm_platform(url: str) -> Optional[str]:
    """
    Check if URL belongs to a DRM-protected platform.
    Returns platform name if DRM, None otherwise.
    """
    domain = _extract_domain(url)
    for drm_domain in DRM_DOMAINS:
        if drm_domain in domain:
            # Amazon special case: only video URLs are DRM, not all of amazon.com
            if 'amazon.com' in drm_domain:
                url_lower = url.lower()
                if any(p in url_lower for p in _AMAZON_VIDEO_PATTERNS):
                    return 'Amazon Prime Video'
                return None  # Not a video URL, let yt-dlp try
            return drm_domain.split('.')[0].capitalize()
    return None


def get_platform_hint(url: str) -> Optional[dict]:
    """Get platform-specific tips for troubleshooting."""
    domain = _extract_domain(url)
    for key, hint in PLATFORM_HINTS.items():
        if key in domain:
            return hint
    return None


def _verify_ytdlp():
    """Ensure yt-dlp is installed."""
    if shutil.which("yt-dlp"):
        return
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


def _is_running_locally() -> bool:
    """Detect if the app is running on localhost vs cloud."""
    # Docker containers / Railway set these
    if os.getenv('RAILWAY_ENVIRONMENT') or os.getenv('RENDER') or os.getenv('FLY_APP_NAME'):
        return False
    # Check if running inside Docker
    if os.path.exists('/.dockerenv'):
        return False
    return True


def _detect_browser() -> Optional[str]:
    """
    Detect which browser is available for cookie extraction.
    Returns browser name for yt-dlp's --cookies-from-browser flag.
    Only works on localhost (same machine as the browser).
    """
    if not _is_running_locally():
        return None

    import platform as plat
    system = plat.system()

    if system == 'Windows':
        # Firefox first — no DPAPI issues, no locked DB issues
        browsers = [
            ('firefox', [
                os.path.expandvars(r'%APPDATA%\Mozilla\Firefox\Profiles'),
            ]),
            ('edge', [
                os.path.expandvars(r'%LOCALAPPDATA%\Microsoft\Edge\User Data'),
            ]),
            ('brave', [
                os.path.expandvars(r'%LOCALAPPDATA%\BraveSoftware\Brave-Browser\User Data'),
            ]),
            ('chrome', [
                os.path.expandvars(r'%LOCALAPPDATA%\Google\Chrome\User Data'),
                os.path.expandvars(r'%PROGRAMFILES%\Google\Chrome\Application\chrome.exe'),
                os.path.expandvars(r'%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe'),
            ]),
        ]
    elif system == 'Darwin':  # macOS
        browsers = [
            ('firefox', [os.path.expanduser('~/Library/Application Support/Firefox/Profiles')]),
            ('safari', [os.path.expanduser('~/Library/Safari')]),
            ('chrome', [os.path.expanduser('~/Library/Application Support/Google/Chrome')]),
            ('brave', [os.path.expanduser('~/Library/Application Support/BraveSoftware/Brave-Browser')]),
        ]
    else:  # Linux
        browsers = [
            ('firefox', [os.path.expanduser('~/.mozilla/firefox')]),
            ('chrome', [os.path.expanduser('~/.config/google-chrome')]),
            ('chromium', [os.path.expanduser('~/.config/chromium')]),
            ('brave', [os.path.expanduser('~/.config/BraveSoftware/Brave-Browser')]),
        ]

    for name, paths in browsers:
        if name in _browser_blacklist:
            continue  # skip previously failed browsers
        for p in paths:
            if os.path.exists(p):
                logger.info(f"Detected browser for cookies: {name}")
                return name

    return None


def _get_cookies_config() -> dict:
    """
    Determine the best cookie strategy.
    Priority:
      1. COOKIES_FILE env var or cookies.txt file → use cookiefile
      2. Local machine with browser → use cookies-from-browser (zero friction)
         Tries multiple browsers in case one fails (e.g. Chrome DPAPI on v127+)
      3. Nothing available → no cookies

    Returns dict with either:
      {'cookiefile': '/path/to/cookies.txt'}
      {'cookiesfrombrowser': 'firefox'}
      {} (empty — no cookies)
    """
    # Priority 1: explicit cookie file
    env_path = os.getenv('COOKIES_FILE')
    if env_path and os.path.isfile(env_path):
        logger.info(f"Using cookie file from env: {env_path}")
        return {'cookiefile': env_path}

    local_path = os.path.join(os.path.dirname(__file__), 'cookies.txt')
    if os.path.isfile(local_path):
        logger.info(f"Using cookie file: {local_path}")
        return {'cookiefile': local_path}

    # Priority 2: auto-extract from browser (localhost only)
    # Try multiple browsers — Chrome 127+ has DPAPI issues, Firefox usually works
    browser = _detect_browser()
    if browser:
        logger.info(f"Will extract cookies from browser: {browser}")
        return {'cookiesfrombrowser': browser}

    # Priority 3: no cookies
    return {}


# Browsers that failed cookie extraction this session
_browser_blacklist = set()

# ALL Chromium-based browsers share the same DPAPI encryption on Windows.
# If one fails DPAPI, they ALL will. Don't waste retries cycling through them.
_CHROMIUM_BROWSERS = {'chrome', 'edge', 'brave', 'chromium', 'opera', 'vivaldi'}


def _blacklist_browser(browser: str, reason: str = ''):
    """
    Blacklist a browser. If DPAPI-related, blacklist ALL Chromium browsers
    since they all use the same Windows encryption and will all fail.
    """
    _browser_blacklist.add(browser)
    dpapi_keywords = ['dpapi', 'failed to decrypt', 'could not copy', 'cookie database']
    if any(kw in reason.lower() for kw in dpapi_keywords) and browser in _CHROMIUM_BROWSERS:
        logger.warning(f"DPAPI error on {browser} — blacklisting ALL Chromium browsers (they all share DPAPI)")
        _browser_blacklist.update(_CHROMIUM_BROWSERS)
    else:
        logger.warning(f"Blacklisting browser: {browser} ({reason})")


def _get_cookies_config_with_fallback() -> dict:
    """
    Like _get_cookies_config but tracks browsers that fail (DPAPI etc.)
    and skips them on subsequent calls within the same process.
    """
    config = _get_cookies_config()
    if 'cookiesfrombrowser' in config:
        browser = config['cookiesfrombrowser']
        if browser in _browser_blacklist:
            # This browser failed before — only Firefox can work if DPAPI is the issue
            alt = _find_non_blacklisted_browser()
            if alt:
                logger.info(f"Browser {browser} blacklisted, using {alt}")
                return {'cookiesfrombrowser': alt}
            else:
                logger.warning(f"All browsers blacklisted, proceeding without cookies")
                return {}
    return config


def _find_non_blacklisted_browser() -> Optional[str]:
    """Find a browser that hasn't been blacklisted. Prefers Firefox (no DPAPI)."""
    if not _is_running_locally():
        return None

    import platform as plat
    system = plat.system()

    # Firefox FIRST and ONLY if DPAPI failed — it's the only browser
    # that uses its own encryption instead of Windows DPAPI
    if system == 'Windows':
        candidates = {
            'firefox': [os.path.expandvars(r'%APPDATA%\Mozilla\Firefox\Profiles')],
        }
    elif system == 'Darwin':
        candidates = {
            'firefox': [os.path.expanduser('~/Library/Application Support/Firefox/Profiles')],
            'safari': [os.path.expanduser('~/Library/Safari')],
        }
    else:
        candidates = {
            'firefox': [os.path.expanduser('~/.mozilla/firefox')],
        }

    for name, paths in candidates.items():
        if name in _browser_blacklist:
            continue
        for p in paths:
            if os.path.exists(p):
                return name
    return None


def check_ytdlp_version() -> dict:
    """Check yt-dlp version and whether an update is available."""
    try:
        import yt_dlp
        current = yt_dlp.version.__version__
        return {'installed': True, 'version': current}
    except ImportError:
        return {'installed': False, 'version': None}
    except Exception:
        return {'installed': True, 'version': 'unknown'}


def get_video_info(url: str) -> dict:
    """Fetch video metadata without downloading."""
    _verify_ytdlp()

    # Check DRM first
    drm = is_drm_platform(url)
    if drm:
        return {
            'title': 'Unknown',
            'duration': 0,
            'url': url,
            'error': f'{drm} uses DRM protection. yt-dlp cannot download DRM-protected content.',
        }

    try:
        import yt_dlp
        is_youtube = any(yt in url.lower() for yt in ['youtube.com', 'youtu.be'])
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'geo_bypass': True,
            'socket_timeout': 30,
        }
        # YouTube: use multiple player clients to bypass bot detection
        # especially for Made-for-Kids / COPPA content
        if is_youtube:
            ydl_opts['extractor_args'] = {
                'youtube': {'player_client': _YT_PLAYER_CLIENTS}
            }

        cookies_config = _get_cookies_config_with_fallback()
        if 'cookiefile' in cookies_config:
            ydl_opts['cookiefile'] = cookies_config['cookiefile']
        elif 'cookiesfrombrowser' in cookies_config:
            ydl_opts['cookiesfrombrowser'] = (cookies_config['cookiesfrombrowser'],)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'description': info.get('description', ''),
                    'thumbnail': info.get('thumbnail', ''),
                    'platform': _detect_platform(url),
                    'url': url,
                }
        except Exception as inner_e:
            # If browser cookie extraction failure, blacklist and retry
            _cookie_errs = ['DPAPI', 'Failed to decrypt', 'Could not copy',
                            'cookie database', 'unable to extract cookies',
                            'permission denied', 'PermissionError',
                            'could not copy', 'locked', 'being used by another']
            if any(kw.lower() in str(inner_e).lower() for kw in _cookie_errs):
                if 'cookiesfrombrowser' in ydl_opts:
                    failed = ydl_opts['cookiesfrombrowser'][0]
                    # Blacklist — if DPAPI, kills ALL Chromium browsers at once
                    _blacklist_browser(failed, str(inner_e))

                    # Only Firefox can survive DPAPI — try it if available
                    alt = _find_non_blacklisted_browser()
                    if alt:
                        logger.info(f"Retrying get_video_info with {alt} (non-Chromium)")
                        ydl_opts['cookiesfrombrowser'] = (alt,)
                        try:
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                info = ydl.extract_info(url, download=False)
                                return {
                                    'title': info.get('title', 'Unknown'),
                                    'duration': info.get('duration', 0),
                                    'uploader': info.get('uploader', 'Unknown'),
                                    'description': info.get('description', ''),
                                    'thumbnail': info.get('thumbnail', ''),
                                    'platform': _detect_platform(url),
                                    'url': url,
                                }
                        except Exception:
                            _blacklist_browser(alt, str(inner_e))
                            logger.warning(f"Firefox also failed, trying without cookies")

                    # Fall back to no cookies
                    ydl_opts.pop('cookiesfrombrowser', None)
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                        return {
                            'title': info.get('title', 'Unknown'),
                            'duration': info.get('duration', 0),
                            'uploader': info.get('uploader', 'Unknown'),
                            'description': info.get('description', ''),
                            'thumbnail': info.get('thumbnail', ''),
                            'platform': _detect_platform(url),
                            'url': url,
                        }
            raise inner_e
    except Exception as e:
        error_msg = str(e)
        hint = get_platform_hint(url)
        if hint:
            error_msg += f"\n\nTip ({hint['name']}): {hint['known_issues']}"
        logger.error(f"Failed to get video info: {error_msg}")
        return {
            'title': 'Unknown',
            'duration': 0,
            'url': url,
            'platform': _detect_platform(url),
            'error': error_msg,
        }


def download_video(url: str, output_dir: str,
                   max_resolution: int = 1080,
                   retries: int = 3,
                   progress_hook=None) -> DownloadResult:
    """
    Download video from any URL using yt-dlp.
    Supports 1800+ platforms. Auto-uses cookies if available.
    Retries on transient failures (network, rate limits).

    Args:
        url: Video URL (any platform yt-dlp supports)
        output_dir: Directory to save the downloaded video
        max_resolution: Maximum video height (default 1080p)
        retries: Number of retry attempts for transient failures
        progress_hook: Optional callback(dict) for download progress updates

    Returns:
        DownloadResult with file path and metadata
    """
    _verify_ytdlp()

    platform = _detect_platform(url)

    # ── DRM check ──
    drm = is_drm_platform(url)
    if drm:
        return DownloadResult(
            success=False,
            platform=platform,
            error=f'{drm} uses DRM protection and cannot be downloaded. '
                  f'This is a hard limitation — no workaround exists.',
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_youtube = any(yt in url.lower() for yt in ['youtube.com', 'youtu.be'])
    logger.info(f"Downloading from {platform}: {url}")

    try:
        import yt_dlp

        # Output template — use video ID only (avoids Unicode filename issues on Windows)
        output_template = str(output_dir / '%(id)s.%(ext)s')

        ffmpeg_available = _has_ffmpeg()

        if ffmpeg_available:
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
                'socket_timeout': 60,
                'retries': retries,
                'fragment_retries': retries,
                'extractor_retries': retries,
                'file_access_retries': 3,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/125.0.0.0 Safari/537.36',
                },
            }
            # Use aria2c for parallel chunk downloads if available (2-5x faster)
            if shutil.which('aria2c'):
                ydl_opts['external_downloader'] = 'aria2c'
                ydl_opts['external_downloader_args'] = ['--min-split-size=1M', '--max-connection-per-server=16', '--split=16']
                logger.info("FFmpeg + aria2c detected — using best quality with parallel downloads")
            else:
                logger.info("FFmpeg detected — using best quality (separate streams + merge)")
        else:
            ydl_opts = {
                'format': f'best[height<={max_resolution}][ext=mp4]/best[height<={max_resolution}]/best',
                'outtmpl': output_template,
                'quiet': False,
                'no_warnings': False,
                'geo_bypass': True,
                'max_filesize': 2 * 1024 * 1024 * 1024,
                'socket_timeout': 60,
                'retries': retries,
                'fragment_retries': retries,
                'extractor_retries': retries,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/125.0.0.0 Safari/537.36',
                },
            }
            logger.warning(
                "FFmpeg NOT found — downloading single combined stream. "
                "Quality may be lower. Install FFmpeg for best results."
            )

        # ── YouTube: use multiple player clients to bypass bot detection ──
        # Critical for Made-for-Kids / COPPA content which triggers bot detection
        # on the default 'web' client, especially on datacenter IPs
        if is_youtube:
            ydl_opts['extractor_args'] = {
                'youtube': {'player_client': _YT_PLAYER_CLIENTS}
            }

        # ── Cookie support (auto-detect best method, with DPAPI fallback) ──
        cookies_config = _get_cookies_config_with_fallback()
        if 'cookiefile' in cookies_config:
            ydl_opts['cookiefile'] = cookies_config['cookiefile']
            logger.info(f"Using cookie file: {cookies_config['cookiefile']}")
        elif 'cookiesfrombrowser' in cookies_config:
            ydl_opts['cookiesfrombrowser'] = (cookies_config['cookiesfrombrowser'],)
            logger.info(f"Extracting cookies from browser: {cookies_config['cookiesfrombrowser']}")

        # ── Progress hook (if provided) ──
        if progress_hook:
            ydl_opts['progress_hooks'] = [progress_hook]

        # ── Download with retry logic ──
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)

                    if info.get('requested_downloads'):
                        file_path = info['requested_downloads'][0]['filepath']
                    else:
                        file_path = ydl.prepare_filename(info)
                        mp4_path = Path(file_path).with_suffix('.mp4')
                        if mp4_path.exists():
                            file_path = str(mp4_path)

                    title = info.get('title', 'Unknown')
                    duration = info.get('duration', 0)

                    logger.info(f"Download complete: {file_path}")
                    logger.info(f"Platform: {platform} | Title: {title} | Duration: {duration}s")

                    warning = None
                    if not cookies_config:
                        hint = get_platform_hint(url)
                        if hint and 'cookies' in hint.get('login_tip', '').lower():
                            warning = (f"Tip: For private {hint['name']} content, "
                                       f"add a cookies.txt file to your project root.")

                    return DownloadResult(
                        success=True,
                        file_path=str(file_path),
                        title=title,
                        duration=duration,
                        platform=platform,
                        warning=warning,
                    )

            except Exception as e:
                last_error = str(e)

                # ── Browser cookie extraction failure → blacklist browser, retry with alternative ──
                # MUST catch Exception (not just DownloadError) because cookie errors
                # from YoutubeDL.__init__() raise PermissionError/OSError, not DownloadError.
                # Covers: DPAPI decryption (Chrome 127+), locked database (Chrome running),
                # permission errors, missing profiles, etc.
                _cookie_fail_keywords = ['DPAPI', 'Failed to decrypt', 'Could not copy',
                                         'cookie database', 'unable to extract cookies',
                                         'permission denied', 'PermissionError',
                                         'could not copy', 'locked', 'being used by another']
                if any(kw.lower() in last_error.lower() for kw in _cookie_fail_keywords):
                    if 'cookiesfrombrowser' in ydl_opts:
                        failed = ydl_opts['cookiesfrombrowser'][0]
                        # Blacklist — if DPAPI, this kills ALL Chromium browsers at once
                        _blacklist_browser(failed, last_error)
                        alt = _find_non_blacklisted_browser()
                        if alt:
                            logger.info(f"Switching to {alt} (non-Chromium, no DPAPI)")
                            ydl_opts['cookiesfrombrowser'] = (alt,)
                            continue
                        else:
                            # No working browser — strip cookies entirely and retry
                            logger.warning("No working browser for cookies, proceeding without")
                            ydl_opts.pop('cookiesfrombrowser', None)
                            continue

                # YouTube bot detection ("Sign in to confirm") — retry with
                # different player client before giving up. Made-for-Kids / COPPA
                # videos trigger this aggressively on the default 'web' client.
                if 'sign in to confirm' in last_error.lower() and is_youtube:
                    current_clients = ydl_opts.get('extractor_args', {}).get('youtube', {}).get('player_client', [])
                    if len(current_clients) > 1:
                        # Drop the first client that failed, try remaining ones
                        remaining = current_clients[1:]
                        logger.warning(f"YouTube bot detection hit, dropping client '{current_clients[0]}', trying {remaining}")
                        ydl_opts['extractor_args'] = {'youtube': {'player_client': remaining}}
                        continue
                    elif not ydl_opts.get('cookiefile') and not ydl_opts.get('cookiesfrombrowser'):
                        # All clients exhausted AND no cookies — this is truly permanent
                        logger.error("YouTube bot detection: all player clients failed and no cookies available")
                        break
                    else:
                        # Has cookies but still failing — permanent
                        break

                # Don't retry on permanent errors
                permanent = ['DRM', 'private', 'not available', 'copyright',
                             'removed', 'terminated', 'does not exist',
                             'Unsupported URL', 'No video formats']
                if any(kw.lower() in last_error.lower() for kw in permanent):
                    break
                if attempt < retries:
                    wait = attempt * 3
                    logger.warning(f"Attempt {attempt}/{retries} failed, retrying in {wait}s: {last_error}")
                    time.sleep(wait)
                continue

        # ── All retries exhausted — build a helpful error ──
        error_msg = last_error or 'Unknown download error'
        hint = get_platform_hint(url)
        if hint:
            error_msg += f"\n\n{hint['name']} tip: {hint['known_issues']}"
            if not cookies_config:
                error_msg += f"\n{hint['login_tip']}"

        logger.error(f"Download failed after {retries} attempts: {error_msg}")
        return DownloadResult(
            success=False,
            platform=platform,
            error=error_msg,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Download failed: {error_msg}")
        return DownloadResult(
            success=False,
            platform=platform,
            error=error_msg,
        )


def get_supported_platforms() -> dict:
    """
    Return a summary of platform support for the UI.
    """
    return {
        'fully_supported': [
            'YouTube', 'YouTube Shorts', 'TikTok', 'Instagram Reels',
            'X/Twitter', 'Facebook (public)', 'Vimeo', 'Dailymotion',
            'Reddit', 'Twitch Clips/VODs', 'Bilibili', 'Rumble',
            'Kick', 'Odysee', 'SoundCloud', 'Internet Archive',
        ],
        'with_cookies': [
            'Instagram (private)', 'Facebook (private)', 'LinkedIn',
            'X/Twitter (restricted)', 'Twitch (sub-only)',
            'YouTube (age-restricted)',
        ],
        'not_supported': [
            'Netflix', 'Disney+', 'Amazon Prime Video', 'Hulu',
            'HBO Max', 'Apple TV+', 'Paramount+', 'Crunchyroll',
        ],
        'total_platforms': '1800+',
        'note': 'Any platform supported by yt-dlp works. This list shows the most common ones.',
    }
