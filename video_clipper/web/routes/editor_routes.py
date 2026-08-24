"""
Editor Routes Blueprint
Endpoints for subtitle generation/burning, text overlays, thumbnail generation, and video modulation.
"""
import os
import uuid
from pathlib import Path
from flask import Blueprint, jsonify, request

from video_clipper.video_processing.subtitle_generator import (
    generate_subtitles, burn_subtitles, burn_text_overlay
)
from video_clipper.video_processing.thumbnail_generator import (
    generate_template_thumbnail, generate_ai_thumbnail, pick_top_frames
)
from video_clipper.video_processing.video_modulator import (
    modulate_video, ModulationConfig, get_presets
)
from video_clipper.web.context import OUTPUT_FOLDER

editor_bp = Blueprint("editor_api", __name__, url_prefix="/api/editor")


@editor_bp.route("/subtitles/generate", methods=["POST"])
def generate_subtitles_endpoint():
    data = request.get_json() or {}
    video_path = data.get("video_path", "").strip()
    model_size = data.get("model_size", "base")
    language = data.get("language") or None

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Valid video_path is required"}), 400

    try:
        res = generate_subtitles(video_path, model_size=model_size, language=language)
        return jsonify({"success": True, "srt": res["srt"], "method": res["method"], "word_count": len(res["words"])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@editor_bp.route("/subtitles/burn", methods=["POST"])
def burn_subtitles_endpoint():
    data = request.get_json() or {}
    video_path = data.get("video_path", "").strip()
    srt_content = data.get("srt", "").strip()

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Valid video_path is required"}), 400
    if not srt_content:
        return jsonify({"error": "SRT content is required"}), 400

    stem = Path(video_path).stem
    out_path = os.path.join(OUTPUT_FOLDER, f"{stem}_subtitled_{str(uuid.uuid4())[:6]}.mp4")

    try:
        saved = burn_subtitles(video_path, srt_content, out_path)
        return jsonify({"success": True, "output_path": saved})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@editor_bp.route("/overlay/burn", methods=["POST"])
def burn_overlay_endpoint():
    data = request.get_json() or {}
    video_path = data.get("video_path", "").strip()
    text_blocks = data.get("blocks", [])

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Valid video_path is required"}), 400
    if not text_blocks:
        return jsonify({"error": "Text blocks array required"}), 400

    stem = Path(video_path).stem
    out_path = os.path.join(OUTPUT_FOLDER, f"{stem}_overlay_{str(uuid.uuid4())[:6]}.mp4")

    try:
        saved = burn_text_overlay(video_path, out_path, text_blocks)
        return jsonify({"success": True, "output_path": saved})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@editor_bp.route("/thumbnail/generate", methods=["POST"])
def generate_thumbnail_endpoint():
    data = request.get_json() or {}
    video_path = data.get("video_path", "").strip()
    title = data.get("title", "").strip()
    mode = data.get("mode", "template")
    style = data.get("style", "bold")

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Valid video_path is required"}), 400

    stem = Path(video_path).stem
    out_path = os.path.join(OUTPUT_FOLDER, f"{stem}_thumb_{str(uuid.uuid4())[:6]}.png")

    try:
        if mode == "ai":
            saved = generate_ai_thumbnail(video_path, title or stem, out_path)
        else:
            saved = generate_template_thumbnail(video_path, title or stem, out_path, style=style)
        return jsonify({"success": True, "thumbnail_path": saved})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@editor_bp.route("/thumbnail/candidates", methods=["GET"])
def get_thumbnail_candidates_endpoint():
    video_path = request.args.get("video_path", "").strip()
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Valid video_path required"}), 400

    candidates = pick_top_frames(video_path, num_candidates=15, top_n=6)
    return jsonify({"candidates": candidates})


@editor_bp.route("/modulate", methods=["POST"])
def modulate_video_endpoint():
    data = request.get_json() or {}
    video_path = data.get("video_path", "").strip()

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Valid video_path required"}), 400

    preset_name = data.get("preset", "")
    presets = get_presets()
    if preset_name in presets:
        cfg = ModulationConfig(**presets[preset_name]["config"])
    else:
        cfg = ModulationConfig(
            zoom_percent=float(data.get("zoom_percent", 4.0)),
            mirror_flip=bool(data.get("mirror_flip", False)),
            color_grade=str(data.get("color_grade", "warm")),
            grain_overlay=bool(data.get("grain_overlay", True)),
            grain_intensity=float(data.get("grain_intensity", 0.05)),
            speed_shift=float(data.get("speed_shift", 1.0)),
        )

    stem = Path(video_path).stem
    out_path = os.path.join(OUTPUT_FOLDER, f"{stem}_mod_{str(uuid.uuid4())[:6]}.mp4")

    res = modulate_video(video_path, out_path, cfg)
    if res.success:
        return jsonify({"success": True, "result": res.to_dict()})
    return jsonify({"error": res.error or "Modulation failed"}), 500


@editor_bp.route("/modulate/presets", methods=["GET"])
def get_modulate_presets_endpoint():
    return jsonify({"presets": get_presets()})
