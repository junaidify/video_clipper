"""
Media Serving Routes Blueprint
Static file server for uploads, generated clips, video library, and factory output.
"""
from flask import Blueprint, send_from_directory

from video_clipper.web.context import (
    UPLOAD_FOLDER, OUTPUT_FOLDER, LIBRARY_FOLDER, FACTORY_FOLDER
)

media_bp = Blueprint("media_serving", __name__)


@media_bp.route("/uploads/<path:filename>")
def serve_upload(filename: str):
    return send_from_directory(UPLOAD_FOLDER, filename)


@media_bp.route("/clips_output/<path:filename>")
def serve_clip(filename: str):
    return send_from_directory(OUTPUT_FOLDER, filename)


@media_bp.route("/video_library/<path:filename>")
def serve_library_file(filename: str):
    return send_from_directory(LIBRARY_FOLDER, filename)


@media_bp.route("/factory_output/<path:filename>")
def serve_factory_file(filename: str):
    return send_from_directory(FACTORY_FOLDER, filename)
