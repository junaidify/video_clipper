"""
YouTube Publishing Routes Blueprint
OAuth2 authentication flows, single clip uploads, batch uploads, and metadata suggestions.
"""
import uuid
from flask import Blueprint, jsonify, request, redirect

from video_clipper.distribution.youtube_uploader import (
    is_configured, is_authenticated, get_auth_url, handle_oauth_callback,
    disconnect, upload_video, generate_clip_metadata
)
from video_clipper.web.context import get_upload_job_state
from video_clipper.web.job_manager import start_youtube_upload_batch

youtube_bp = Blueprint("youtube_api", __name__)


@youtube_bp.route("/api/youtube/status", methods=["GET"])
def youtube_status():
    return jsonify({
        "configured": is_configured(),
        "authenticated": is_authenticated(),
    })


@youtube_bp.route("/api/youtube/auth", methods=["GET"])
@youtube_bp.route("/api/youtube/auth-url", methods=["GET"])
def youtube_auth_url():
    if not is_configured():
        return jsonify({"error": "Google Client ID / Secret not configured in .env"}), 400

    redirect_uri = request.host_url.rstrip("/") + "/oauth2callback"
    try:
        url = get_auth_url(redirect_uri)
        return jsonify({"auth_url": url, "url": url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@youtube_bp.route("/oauth2callback", methods=["GET"])
def oauth2callback():
    code = request.args.get("code")
    if not code:
        return "Authentication cancelled or missing code", 400

    redirect_uri = request.host_url.rstrip("/") + "/oauth2callback"
    try:
        handle_oauth_callback(code, redirect_uri)
        return "<script>window.opener ? window.opener.postMessage('youtube_auth_success', '*') : null; window.close();</script><h3>Connected to YouTube! You can close this window.</h3>"
    except Exception as e:
        return f"<h3>Authentication failed: {e}</h3>", 500


@youtube_bp.route("/api/youtube/disconnect", methods=["POST"])
def youtube_disconnect():
    disconnect()
    return jsonify({"success": True})


@youtube_bp.route("/api/youtube/upload", methods=["POST"])
@youtube_bp.route("/api/youtube/upload-single", methods=["POST"])
def upload_single_video():
    if not is_authenticated():
        return jsonify({"error": "Not authenticated with YouTube."}), 401

    data = request.get_json() or {}
    file_path = data.get("file_path", "").strip() or data.get("clip_path", "").strip()
    title = data.get("title", "").strip()

    if not file_path:
        return jsonify({"error": "file_path is required"}), 400

    res = upload_video(
        file_path=file_path,
        title=title or "Video Clip",
        description=data.get("description", ""),
        tags=data.get("tags", []),
        category=data.get("category", "entertainment"),
        privacy=data.get("privacy", "private"),
        made_for_kids=data.get("made_for_kids", False),
        publish_at=data.get("publish_at"),
    )

    return jsonify(res.__dict__)


@youtube_bp.route("/api/youtube/upload-batch", methods=["POST"])
def upload_batch_videos():
    if not is_authenticated():
        return jsonify({"error": "Not authenticated with YouTube."}), 401

    data = request.get_json() or {}
    clips = data.get("clips", [])
    if not clips:
        return jsonify({"error": "No clips provided"}), 400

    job_id = str(uuid.uuid4())[:8]
    start_youtube_upload_batch(
        job_id=job_id,
        clips=clips,
        source_title=data.get("source_title", ""),
        category=data.get("category", "entertainment"),
        privacy=data.get("privacy", "private"),
        made_for_kids=data.get("made_for_kids", False),
        schedule_interval_hours=float(data.get("schedule_interval_hours", 0.0)),
    )

    return jsonify({"job_id": job_id, "status": "uploading"})


@youtube_bp.route("/api/youtube/upload-status/<job_id>", methods=["GET"])
def upload_status(job_id: str):
    state = get_upload_job_state(job_id)
    if not state:
        return jsonify({"error": "Upload job not found"}), 404
    return jsonify(state)


@youtube_bp.route("/api/youtube/suggest-metadata", methods=["POST"])
@youtube_bp.route("/api/youtube/generate-metadata", methods=["POST"])
def suggest_metadata():
    data = request.get_json() or {}
    clip_info = data.get("clip", {})
    source_title = data.get("source_title", "")
    clip_number = int(data.get("clip_number", 1))
    total_clips = int(data.get("total_clips", 1))

    meta = generate_clip_metadata(clip_info, source_title, clip_number, total_clips)
    return jsonify(meta)


@youtube_bp.route("/api/youtube/generate-metadata-batch", methods=["POST"])
def suggest_metadata_batch():
    data = request.get_json() or {}
    clips = data.get("clips", [])
    source_title = data.get("source_title", "")
    results = []
    for i, clip in enumerate(clips, 1):
        results.append(generate_clip_metadata(clip, source_title, i, len(clips)))
    return jsonify({"metadata": results})
