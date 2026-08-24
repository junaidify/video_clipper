"""
Library API Routes Blueprint
CRUD endpoints for video library management and persistent asset storage.
"""
import os
from pathlib import Path
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from video_clipper.media_management.downloader import download_video
from video_clipper.web.context import get_video_library, UPLOAD_FOLDER

library_bp = Blueprint("library_api", __name__, url_prefix="/api/library")


@library_bp.route("", methods=["GET"])
def list_library_videos():
    """List all videos in the persistent library."""
    lib = get_video_library()
    videos = [v.to_dict() for v in lib.list_videos()]
    return jsonify({"videos": videos, "count": len(videos)})


@library_bp.route("/add-upload", methods=["POST"])
def add_uploaded_to_library():
    """Upload a video file directly into the persistent library."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    title = request.form.get("title", "").strip() or Path(file.filename).stem
    safe_name = secure_filename(file.filename)
    tmp_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(tmp_path)

    lib = get_video_library()
    entry = lib.add_video(
        source_path=tmp_path,
        title=title,
        source="upload",
    )

    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return jsonify({"success": True, "video": entry.to_dict()})


@library_bp.route("/download", methods=["POST"])
def download_to_library():
    """Download video from URL directly into persistent library."""
    data = request.get_json() or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL is required"}), 400

    dl_result = download_video(url, output_dir=UPLOAD_FOLDER)
    if not dl_result.success or not dl_result.file_path:
        return jsonify({"error": dl_result.error or "Download failed"}), 400

    lib = get_video_library()
    entry = lib.add_video(
        source_path=dl_result.file_path,
        title=dl_result.title or "Downloaded Video",
        source="url",
        source_url=url,
        duration=dl_result.duration,
        uploader=dl_result.uploader,
        channel_url=dl_result.channel_url,
    )

    return jsonify({"success": True, "video": entry.to_dict()})


@library_bp.route("/<video_id>", methods=["GET"])
def get_library_video(video_id: str):
    """Retrieve details for a single video in the library."""
    lib = get_video_library()
    entry = lib.get_video(video_id)
    if not entry:
        return jsonify({"error": "Video not found"}), 404
    return jsonify({"video": entry.to_dict()})


@library_bp.route("/<video_id>", methods=["DELETE"])
def delete_library_video(video_id: str):
    """Delete video and all its files from the library."""
    lib = get_video_library()
    deleted = lib.delete_video(video_id)
    if not deleted:
        return jsonify({"error": "Video not found or already deleted"}), 404
    return jsonify({"success": True})


@library_bp.route("/<video_id>/clips", methods=["GET"])
def get_clips_for_library_video(video_id: str):
    """List all clip output directories associated with a video."""
    lib = get_video_library()
    entry = lib.get_video(video_id)
    if not entry:
        return jsonify({"error": "Video not found"}), 404

    dirs = lib.get_clips_for_video(video_id)
    sessions = []
    for d in dirs:
        clips = []
        p = Path(d)
        if p.is_dir():
            for f in sorted(p.glob("clip_*.mp4")):
                size_mb = round(f.stat().st_size / (1024 * 1024), 2)
                clips.append({
                    "filename": f.name,
                    "file_path": str(f),
                    "size_mb": size_mb,
                })
            sessions.append({
                "directory": d,
                "session_name": p.name,
                "clips": clips,
                "clip_count": len(clips),
            })

    return jsonify({"sessions": sessions, "total_sessions": len(sessions)})


@library_bp.route("/stats", methods=["GET"])
def get_library_statistics():
    """Return library storage and video counts."""
    lib = get_video_library()
    return jsonify(lib.get_library_stats())
