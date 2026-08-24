"""
YouTube Upload Module
Handles OAuth2 authentication and video uploads to YouTube via Google Data API v3.
Supports resumable uploads, scheduled publishing, subtitle captions, and batch queues.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List

logger = logging.getLogger(__name__)

YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
YOUTUBE_UPLOAD_SCOPE = 'https://www.googleapis.com/auth/youtube.upload'
YOUTUBE_FORCE_SSL_SCOPE = 'https://www.googleapis.com/auth/youtube.force-ssl'
SCOPES = [YOUTUBE_UPLOAD_SCOPE, YOUTUBE_FORCE_SSL_SCOPE]

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

# OAuth token path
TOKEN_FILE = os.path.join(os.getcwd(), '.youtube_token.json')


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
    results: List[dict] = field(default_factory=list)
    error: Optional[str] = None


def _get_client_config() -> dict:
    client_id = os.getenv('GOOGLE_CLIENT_ID', '').strip()
    client_secret = os.getenv('GOOGLE_CLIENT_SECRET', '').strip()
    if not client_id or not client_secret:
        raise ValueError(
            "YouTube upload requires GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET "
            "in your .env file."
        )
    return {
        'web': {
            'client_id': client_id,
            'client_secret': client_secret,
            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
            'token_uri': 'https://oauth2.googleapis.com/token',
            'redirect_uris': [],
        }
    }


def is_configured() -> bool:
    cid = os.getenv('GOOGLE_CLIENT_ID', '').strip()
    csecret = os.getenv('GOOGLE_CLIENT_SECRET', '').strip()
    return bool(cid and csecret)


def is_authenticated() -> bool:
    if not os.path.isfile(TOKEN_FILE):
        return False
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        if creds.valid:
            return True
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                with open(TOKEN_FILE, 'w') as f:
                    f.write(creds.to_json())
                return True
            except Exception:
                try:
                    os.remove(TOKEN_FILE)
                except OSError:
                    pass
                return False
        return False
    except Exception:
        return False


_pending_flow = None


def get_auth_url(redirect_uri: str) -> str:
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
    global _pending_flow

    if _pending_flow is None:
        from google_auth_oauthlib.flow import Flow
        config = _get_client_config()
        config['web']['redirect_uris'] = [redirect_uri]
        _pending_flow = Flow.from_client_config(config, scopes=SCOPES, redirect_uri=redirect_uri)

    _pending_flow.fetch_token(code=auth_code)
    creds = _pending_flow.credentials
    _pending_flow = None

    with open(TOKEN_FILE, 'w') as f:
        f.write(creds.to_json())
    logger.info("YouTube OAuth tokens saved successfully.")
    return True


def disconnect():
    if os.path.isfile(TOKEN_FILE):
        os.remove(TOKEN_FILE)
        logger.info("YouTube token removed.")


def _get_authenticated_service():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    if not os.path.isfile(TOKEN_FILE):
        raise RuntimeError("Not authenticated. Connect YouTube account first.")

    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(TOKEN_FILE, 'w') as f:
                f.write(creds.to_json())
        except Exception as e:
            try:
                os.remove(TOKEN_FILE)
            except OSError:
                pass
            raise RuntimeError(f"YouTube session expired: {e}")

    if not creds.valid:
        try:
            os.remove(TOKEN_FILE)
        except OSError:
            pass
        raise RuntimeError("YouTube credentials invalid. Please re-authenticate.")

    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=creds)


def upload_video(
    file_path: str,
    title: str,
    description: str = '',
    tags: Optional[List[str]] = None,
    category: str = 'entertainment',
    privacy: str = 'private',
    made_for_kids: bool = False,
    publish_at: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> UploadResult:
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    filename = Path(file_path).name

    if not os.path.isfile(file_path):
        return UploadResult(filename=filename, success=False, error='File not found')

    try:
        youtube = _get_authenticated_service()

        title = title[:100]
        description = (description or '')[:5000]
        tags = (tags or [])[:500]
        category_id = str(CATEGORIES.get(category.lower(), 24))

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
            chunksize=1024 * 1024 * 5,
        )

        insert_request = youtube.videos().insert(
            part=','.join(body.keys()),
            body=body,
            media_body=media,
        )

        response = None
        while response is None:
            status, response = insert_request.next_chunk()
            if status and progress_callback:
                bytes_sent = int(status.resumable_progress)
                progress_callback(bytes_sent, file_size)

        video_id = response['id']
        video_url = f'https://youtu.be/{video_id}'
        logger.info(f"Uploaded '{filename}' -> {video_url}")

        return UploadResult(
            filename=filename,
            success=True,
            video_id=video_id,
            video_url=video_url,
        )

    except HttpError as e:
        error_msg = str(e)
        try:
            err_data = json.loads(e.content.decode())
            reason = err_data['error']['errors'][0]['reason']
            message = err_data['error']['errors'][0]['message']
            error_msg = f'{reason}: {message}'
        except Exception:
            pass
        return UploadResult(filename=filename, success=False, error=error_msg)

    except Exception as e:
        return UploadResult(filename=filename, success=False, error=str(e))


def upload_caption(
    video_id: str,
    srt_path: str,
    language: str = 'en',
    name: str = 'English',
) -> bool:
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    if not os.path.isfile(srt_path):
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
    hook = clip_info.get('hook_text', '').strip()
    reason = clip_info.get('reason', '')
    duration = clip_info.get('duration', 0)

    if hook:
        title = hook.split('.')[0].split('!')[0].split('?')[0]
        if len(title) > 90:
            title = title[:87] + '...'
        elif len(title) < 10:
            title = hook[:90]
    else:
        title = f'{source_title} - Clip {clip_number}' if source_title else f'Clip {clip_number}'

    desc_parts = []
    if source_title:
        desc_parts.append(f'From: {source_title}')
    if hook and len(hook) > len(title):
        desc_parts.append(f'\n{hook}')
    desc_parts.append(f'\nClip {clip_number}/{total_clips}')
    desc_parts.append(f'Duration: {int(duration)}s')
    desc_parts.append('\n#Shorts')
    description = '\n'.join(desc_parts)

    tags = ['shorts', 'clips', 'highlights']
    if source_title:
        words = [w.strip('.,!?#') for w in source_title.split() if len(w) > 3]
        tags.extend(words[:10])
    if reason:
        tags.append(reason.replace('_', ' '))

    return {
        'title': title,
        'description': description,
        'tags': list(set(tags))[:15],
    }
