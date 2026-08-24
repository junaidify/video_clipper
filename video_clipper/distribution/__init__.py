"""
Distribution module: social publishing, batch uploads, and platform integrations.
"""
from video_clipper.distribution.youtube_uploader import (
    UploadResult,
    UploadJob,
    is_configured,
    is_authenticated,
    get_auth_url,
    handle_oauth_callback,
    disconnect,
    upload_video,
    upload_caption,
    generate_clip_metadata,
)

__all__ = [
    "UploadResult",
    "UploadJob",
    "is_configured",
    "is_authenticated",
    "get_auth_url",
    "handle_oauth_callback",
    "disconnect",
    "upload_video",
    "upload_caption",
    "generate_clip_metadata",
]
