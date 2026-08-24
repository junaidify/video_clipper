"""
Media Management module: universal multi-platform video downloading and persistent library storage.
"""
from video_clipper.media_management.downloader import (
    DownloadResult,
    download_video,
    get_video_info,
    is_valid_url,
    is_drm_platform,
)
from video_clipper.media_management.library import (
    VideoEntry,
    VideoLibrary,
)

__all__ = [
    "DownloadResult",
    "download_video",
    "get_video_info",
    "is_valid_url",
    "is_drm_platform",
    "VideoEntry",
    "VideoLibrary",
]
