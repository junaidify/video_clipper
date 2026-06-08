"""
YouTube Upload Module
Handles OAuth2 authentication and video uploads to YouTube via Data API v3.

Features:
- OAuth2 flow with refresh token persistence
- Batch upload multiple clips
- Auto-generate title, description, tags
- Upload subtitles/captions
- Resumable uploads with progress tracking
- Privacy control (public/unlisted/private)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# ─── Constants ───
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
YOUTUBE_UPLOAD_SCOPE = 'https://www.googleapis.com/auth/youtube.upload'
YOUTUBE_FORCE_SSL_SCOPE = 'https://www.googleapis.com/auth/youtube.force-ssl'
SCOPES = [YOUTUBE_UPLOAD_SCOPE, YOUTUBE_FORCE_SSL_SCOPE]

# YouTube category IDs
CATEGORIES = {
    'film':          1,
    'autos':         2,
    'music':        10,
    'pets':         15,
    'sports':       17,
    'gaming':       20,
    'people':       22,
    'comedy':       23,
    'entertainment':24,
    'news':         25,
    'howto':        26,
    'education':    27,
    'science':      28,
    'nonprofit':    29,
}

# Where we persist the OAuth refresh token
TOKEN_FILE = os.path.join(os.path.dirname(__file__), '.youtube_token.json')


@dataclass
class UploadResult:
    """Result of a single video upload."""
    filename: str
    success: bool
    video_id: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None


@dataclass
class UploadJob:
    """Tracks a batch upload job."""
    job_id: str
    status: str = 'pending'       # pending | uploading | completed | error
    total: int = 0
    uploaded: int = 0
    current_file: str = ''
    progress_pct: int = 0
    results: list = field(default_factory=list)
    error: Optional[str] = None


def _get_client_config() -> dict:
    """Build OAuth client config from environment variables."""
    client_id = os.getenv('GOOGLE_CLIENT_ID', '').strip()
    client_secret = os.getenv('GOOGLE_CLIENT_SECRET', '').strip()
    if not client_id or not client_secret:
        raise ValueError(
            "YouTube upload requires GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET "
            "in your .env file. Get them from Google Cloud Console → APIs & Services → Credentials."
        )
    return {
        'web': {
            'client_id': client_id,
            'client_secret': client_secret,
            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
            'token_uri': 'https://oauth2.googleapis.com/token',
            'redirect_uris': [],  # set dynamically
        }
    }


def is_configured() -> bool:
    """Check if YouTube upload credentials are configured."""
    cid = os.getenv('GOOGLE_CLIENT_ID', '').strip()
    csecret = os.getenv('GOOGLE_CLIENT_SECRET', '').strip()
    return bool(cid and csecret)


def is_authenticated() -> bool:
    """Check if we have a valid (or refreshable) token."""
    if not os.path.isfile(TOKEN_FILE):
        return False
    try:
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        return creds.valid or (creds.expired and creds.refresh_token)
    except Exception:
        return False


# Module-level flow storage (needed to preserve PKCE code_verifier between auth URL and callback)
_pending_flow = None


def get_auth_url(redirect_uri: str) -> str:
    """
    Generate the Google OAuth2 authorization URL.
    The user visits this URL to grant YouTube upload permission.
    """
    global _pending_flow
    from google_auth_oauthlib.flow import Flow

    config = _get_client_config()
    config['web']['redirect_uris'] = [redirect_uri]

    _pending_flow = Flow.from_client_config(config, scopes=SCOPES, redirect_uri=redirect_uri)
    auth_url, _ = _pending_flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent select_account',
    )
    return auth_url


def handle_oauth_callback(auth_code: str, redirect_uri: str) -> bool:
    """
    Exchange the authorization code for tokens and persist them.
    Returns True on success.
    """
    global _pending_flow

    if _pending_flow is None:
        # Fallback: create new flow without PKCE if the original was lost (e.g. server restart)
        from google_auth_oauthlib.flow import Flow
        config = _get_client_config()
        config['web']['redirect_uris'] = [redirect_uri]
        _pending_flow = Flow.from_client_config(config, scopes=SCOPES, redirect_uri=redirect_uri)

    _pending_flow.fetch_token(code=auth_code)

    creds = _pending_flow.credentials
    _pending_flow = None  # Clear after use
    # Persist tokens
    with open(TOKEN_FILE, 'w') as f:
        f.write(creds.to_json())
    logger.info("YouTube OAuth tokens saved successfully")
    return True


def disconnect():
    """Remove stored tokens (sign out)."""
    if os.path.isfile(TOKEN_FILE):
        os.remove(TOKEN_FILE)
        logger.info("YouTube tokens removed")


def _get_authenticated_service():
    """Build an authenticated YouTube API service object."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    if not os.path.isfile(TOKEN_FILE):
        raise RuntimeError("Not authenticated. Connect YouTube account first.")

    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Re-persist refreshed token
        with open(TOKEN_FILE, 'w') as f:
            f.write(creds.to_json())

    if not creds.valid:
        raise RuntimeError("YouTube credentials invalid. Re-connect your account.")

    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=creds)


def upload_video(
    file_path: str,
    title: str,
    description: str = '',
    tags: list = None,
    category: str = 'entertainment',
    privacy: str = 'private',
    made_for_kids: bool = False,
    publish_at: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> UploadResult:
    """
    Upload a single video to YouTube.

    Args:
        file_path: Path to the video file
        title: Video title (max 100 chars)
        description: Video description (max 5000 chars)
        tags: List of tags
        category: Category name (see CATEGORIES dict)
        privacy: 'public', 'unlisted', or 'private'
        made_for_kids: COPPA setting
        publish_at: ISO 8601 datetime to schedule publish (e.g. '2025-06-01T14:00:00Z').
                     When set, video uploads as private and auto-publishes at this time.
        progress_callback: fn(bytes_sent, total_bytes) called during upload

    Returns:
        UploadResult with video_id and URL on success
    """
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    filename = Path(file_path).name

    if not os.path.isfile(file_path):
        return UploadResult(filename=filename, success=False, error='File not found')

    try:
        youtube = _get_authenticated_service()

        # Truncate title/description to YouTube limits
        title = title[:100]
        description = (description or '')[:5000]
        tags = (tags or [])[:500]  # YouTube allows up to 500 chars total in tags
        category_id = str(CATEGORIES.get(category.lower(), 24))  # default: entertainment

        # Scheduling: YouTube requires privacyStatus='private' + publishAt datetime
        status_body = {
            'privacyStatus': 'private' if publish_at else privacy,
            'selfDeclaredMadeForKids': made_for_kids,
        }
        if publish_at:
            status_body['publishAt'] = publish_at

        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags,
                'categoryId': category_id,
            },
            'status': status_body,
        }

        file_size = os.path.getsize(file_path)
        media = MediaFileUpload(
            file_path,
            mimetype='video/mp4',
            resumable=True,
            chunksize=1024 * 1024 * 5,  # 5 MB chunks
        )

        insert_request = youtube.videos().insert(
            part=','.join(body.keys()),
            body=body,
            media_body=media,
        )

        # Resumable upload loop
        response = None
        while response is None:
            status, response = insert_request.next_chunk()
            if status and progress_callback:
                bytes_sent = int(status.resumable_progress)
                progress_callback(bytes_sent, file_size)

        video_id = response['id']
        video_url = f'https://youtu.be/{video_id}'
        logger.info(f"Uploaded '{filename}' → {video_url}")

        return UploadResult(
            filename=filename,
            success=True,
            video_id=video_id,
            video_url=video_url,
        )

    except HttpError as e:
        error_msg = str(e)
        # Parse the JSON error for a cleaner message
        try:
            err_data = json.loads(e.content.decode())
            reason = err_data['error']['errors'][0]['reason']
            message = err_data['error']['errors'][0]['message']
            error_msg = f'{reason}: {message}'
        except Exception:
            pass
        logger.error(f"YouTube upload failed for '{filename}': {error_msg}")
        return UploadResult(filename=filename, success=False, error=error_msg)

    except Exception as e:
        logger.exception(f"YouTube upload failed for '{filename}'")
        return UploadResult(filename=filename, success=False, error=str(e))


def upload_caption(
    video_id: str,
    srt_path: str,
    language: str = 'en',
    name: str = 'English',
) -> bool:
    """Upload an SRT caption file to an already-uploaded YouTube video."""
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    if not os.path.isfile(srt_path):
        logger.warning(f"SRT file not found: {srt_path}")
        return False

    try:
        youtube = _get_authenticated_service()
        media = MediaFileUpload(srt_path, mimetype='application/x-subrip', resumable=True)

        youtube.captions().insert(
            part='snippet',
            body={
                'snippet': {
                    'videoId': video_id,
                    'language': language,
                    'name': name,
                }
            },
            media_body=media,
        ).execute()

        logger.info(f"Caption uploaded for video {video_id}")
        return True

    except HttpError as e:
        logger.error(f"Caption upload failed for {video_id}: {e}")
        return False


def generate_clip_metadata(
    clip_info: dict,
    source_title: str = '',
    clip_number: int = 1,
    total_clips: int = 1,
) -> dict:
    """
    Auto-generate YouTube metadata for a clip.

    Args:
        clip_info: Dict with keys like 'hook_text', 'reason', 'score', 'duration', etc.
        source_title: Original video title
        clip_number: This clip's number
        total_clips: Total clips being uploaded

    Returns:
        Dict with 'title', 'description', 'tags'
    """
    hook = clip_info.get('hook_text', '').strip()
    reason = clip_info.get('reason', '')
    duration = clip_info.get('duration', 0)
    score = clip_info.get('score', 0)

    # Title: use hook text, truncated nicely
    if hook:
        # Take first sentence or first 90 chars
        title = hook.split('.')[0].split('!')[0].split('?')[0]
        if len(title) > 90:
            title = title[:87] + '...'
        elif len(title) < 10:
            title = hook[:90]
    else:
        title = f'{source_title} - Clip {clip_number}' if source_title else f'Clip {clip_number}'

    # Description
    desc_parts = []
    if source_title:
        desc_parts.append(f'From: {source_title}')
    if hook and len(hook) > len(title):
        desc_parts.append(f'\n{hook}')
    desc_parts.append(f'\nClip {clip_number}/{total_clips}')
    desc_parts.append(f'Duration: {int(duration)}s')
    desc_parts.append('\n#Shorts')
    description = '\n'.join(desc_parts)

    # Tags
    tags = ['shorts', 'clips', 'highlights']
    if source_title:
        # Add words from source title as tags
        words = [w.strip('.,!?#') for w in source_title.split() if len(w) > 3]
        tags.extend(words[:10])
    if reason:
        tags.append(reason.replace('_', ' '))

    return {
        'title': title,
        'description': description,
        'tags': list(set(tags))[:15],  # YouTube recommends max ~15 tags
    }
