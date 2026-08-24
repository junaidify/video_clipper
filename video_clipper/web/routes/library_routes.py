"""
Library API Routes Blueprint
CRUD endpoints for video library management, video URL downloading, and persistent asset storage.
"""
import os
import uuid
from pathlib import Path
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from video_clipper.media_management.downloader import download_video
from video_clipper.web.context import (
    get_video_library,
    UPLOAD_FOLDER,
    get_job_state,
    set_job,
    update_job,
    GLOBAL_THREAD_POOL,
)

library_bp = Blueprint("library_api", __name__, url_prefix="/api")


@library_bp.route("/library", methods=["GET"])
@library_bp.route("/library/list", methods=["GET"])
def list_library_videos():
    """List all videos in the persistent library."""
    lib = get_video_library()
    videos = [v.to_dict() for v in lib.list_videos()]
    return jsonify({"videos": videos, "count": len(videos)})


@library_bp.route("/library/add", methods=["POST"])
def add_to_library():
    """
    Unified entry point to add a video to the library.
    Handles both file uploads and URL downloads with progress polling.
    """
    url = request.form.get("url", "").strip() or (request.json.get("url", "").strip() if request.is_json else "")
    
    if url:
        job_id = f"lib_{str(uuid.uuid4())[:8]}"
        set_job(job_id, {
            "status": "starting",
            "progress": 5,
            "message": "Initializing download...",
            "error": None,
            "video": None,
        })

        def _bg_download(jid: str, download_url: str):
            try:
                def _prog(pct, msg):
                    update_job(jid, {"progress": round(pct, 1), "message": msg})

                dl_res = download_video(
                    download_url,
                    output_dir=UPLOAD_FOLDER,
                    progress_callback=_prog,
                )

                if not dl_res.success or not dl_res.file_path or not os.path.isfile(dl_res.file_path):
                    update_job(jid, {
                        "status": "error",
                        "error": dl_res.error or "Download failed",
                    })
                    return

                lib = get_video_library()
                entry = lib.add_video(
                    source_path=dl_res.file_path,
                    title=dl_res.title or "Downloaded Video",
                    source="url",
                    source_url=download_url,
                    duration=dl_res.duration,
                    uploader=dl_res.uploader,
                    channel_url=dl_res.channel_url,
                    thumbnail_url=dl_res.thumbnail_url,
                )

                try:
                    os.remove(dl_res.file_path)
                except OSError:
                    pass

                update_job(jid, {
                    "status": "completed",
                    "progress": 100,
                    "message": "Video added to library",
                    "video": entry.to_dict(),
                })
            except Exception as e:
                update_job(jid, {
                    "status": "error",
                    "error": str(e),
                })

        GLOBAL_THREAD_POOL.submit(_bg_download, job_id, url)
        return jsonify({"job_id": job_id, "status": "processing"})

    # File upload handling
    if "video" in request.files:
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

    return jsonify({"error": "Provide either a video file or a URL"}), 400


@library_bp.route("/library/add-upload", methods=["POST"])
def add_uploaded_to_library():
    """Upload a video file directly into the persistent library."""
    return add_to_library()


@library_bp.route("/library/download", methods=["POST"])
def download_to_library():
    """Download video from URL directly into persistent library."""
    return add_to_library()


@library_bp.route("/library/<video_id>", methods=["GET"])
def get_library_video(video_id: str):
    """Retrieve details for a single video in the library."""
    lib = get_video_library()
    entry = lib.get_video(video_id)
    if not entry:
        return jsonify({"error": "Video not found"}), 404
    return jsonify({"video": entry.to_dict()})


@library_bp.route("/library/delete", methods=["POST", "DELETE"])
@library_bp.route("/library/<video_id>", methods=["DELETE"])
def delete_library_video(video_id: str = None):
    """Delete video and all its files from the library."""
    if not video_id:
        video_id = request.form.get("video_id") or request.args.get("video_id")
        if not video_id and request.is_json:
            video_id = request.json.get("video_id")

    if not video_id:
        return jsonify({"error": "video_id is required"}), 400

    lib = get_video_library()
    deleted = lib.delete_video(video_id)
    if not deleted:
        return jsonify({"error": "Video not found or already deleted"}), 404
    return jsonify({"success": True})


@library_bp.route("/library/<video_id>/clips", methods=["GET"])
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


@library_bp.route("/library/stats", methods=["GET"])
def get_library_statistics():
    """Return library storage and video counts."""
    lib = get_video_library()
    return jsonify(lib.get_library_stats())
