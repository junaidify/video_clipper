"""
System Routes Blueprint
System health checks, diagnostics, available hardware/LLM providers, cookie management, and URL checking.
"""
import os
import json
import shutil
from pathlib import Path
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from video_clipper.clipping.transcriber import resolve_device, _get_whisper_cache_dir
from video_clipper.clipping.llm_analyzer import LLMConfig
from video_clipper.media_management.downloader import (
    is_valid_url, is_drm_platform, get_video_info, PLATFORM_HINTS, _get_cookie_sources
)
from video_clipper.web.context import UPLOAD_FOLDER

system_bp = Blueprint("system_api", __name__, url_prefix="/api")

# Memory store for upload status if file doesn't exist
_uploaded_clip_map = {}


@system_bp.route("/settings", methods=["GET"])
def get_system_settings():
    llm = LLMConfig.from_env()

    whisper_cache = _get_whisper_cache_dir()
    installed_models = []
    if os.path.isdir(whisper_cache):
        for f in os.listdir(whisper_cache):
            if f.endswith(".pt"):
                installed_models.append(f.replace(".pt", ""))

    ffmpeg_available = bool(shutil.which("ffmpeg"))
    ffprobe_available = bool(shutil.which("ffprobe"))

    cookie_sources = _get_cookie_sources()
    cookie_info = [{"type": s[0], "value": s[1]} for s in cookie_sources]

    return jsonify({
        "ffmpeg": {
            "available": ffmpeg_available,
            "ffprobe_available": ffprobe_available,
        },
        "device": resolve_device("auto"),
        "whisper": {
            "installed_models": installed_models,
            "cache_dir": whisper_cache,
        },
        "llm": {
            "gemini": bool(llm.gemini_api_key),
            "groq": bool(llm.groq_api_key),
            "nvidia": bool(llm.nvidia_api_key),
            "preferred": llm.preferred_provider,
        },
        "cookies": {
            "configured_sources": cookie_info,
            "count": len(cookie_info),
        },
        "integrations": {
            "pexels": bool(os.environ.get("PEXELS_API_KEY")),
            "pixabay": bool(os.environ.get("PIXABAY_API_KEY")),
        }
    })


@system_bp.route("/platforms", methods=["GET"])
def get_supported_platforms():
    return jsonify({"platforms": PLATFORM_HINTS})


@system_bp.route("/cookie-status", methods=["GET"])
def get_cookie_status():
    sources = _get_cookie_sources()
    active_sources = [{"type": s[0], "value": s[1]} for s in sources]
    has_cookies = len(active_sources) > 0
    return jsonify({
        "has_cookies": has_cookies,
        "sources": active_sources,
        "count": len(active_sources)
    })


@system_bp.route("/upload-cookies", methods=["POST"])
def upload_cookie_file():
    if "cookie_file" in request.files:
        file = request.files["cookie_file"]
        if file.filename:
            target = os.path.join(os.getcwd(), "cookies.txt")
            file.save(target)
            return jsonify({"success": True, "message": "Cookie file uploaded successfully."})

    raw_text = request.form.get("cookie_text", "").strip()
    if raw_text:
        target = os.path.join(os.getcwd(), "cookies.txt")
        with open(target, "w", encoding="utf-8") as f:
            f.write(raw_text)
        return jsonify({"success": True, "message": "Cookies saved successfully."})

    return jsonify({"error": "No cookie file or text provided"}), 400


@system_bp.route("/upload-status", methods=["GET", "POST"])
def manage_upload_status():
    global _uploaded_clip_map
    if request.method == "POST":
        data = request.get_json() or {}
        clip_id = data.get("clip_id")
        status = data.get("status", True)
        if clip_id:
            _uploaded_clip_map[clip_id] = status
        return jsonify({"success": True})
    return jsonify(_uploaded_clip_map)


@system_bp.route("/check-url", methods=["POST"])
def probe_url():
    data = request.get_json() or {}
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"valid": False, "error": "Empty URL"}), 400

    if not is_valid_url(url):
        return jsonify({"valid": False, "error": "Invalid URL format"}), 400

    drm = is_drm_platform(url)
    if drm:
        return jsonify({
            "valid": False,
            "is_drm": True,
            "platform": drm,
            "error": f"{drm} uses DRM encryption and cannot be downloaded.",
        })

    info = get_video_info(url)
    if not info:
        return jsonify({"valid": False, "error": "Could not inspect video URL"}), 400

    return jsonify({"valid": True, "info": info})
