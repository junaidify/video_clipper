"""
Factory Routes Blueprint
Endpoints for Automated Shorts Content Generation, Trend Scouting, Story Customizer, and Asset Synthesis.
"""
import os
import uuid
from pathlib import Path
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from video_clipper.content_generation.trend_scout import (
    scout_trending, get_available_categories
)
from video_clipper.content_generation.script_generator import (
    generate_script, get_style_presets
)
from video_clipper.content_generation.factory_orchestrator import (
    start_generation, start_custom_generation, get_job, list_jobs, cleanup_job
)
from video_clipper.audio_processing.tts_engine import get_available_voices
from video_clipper.audio_processing.music_mixer import get_mood_options
from video_clipper.web.context import UPLOAD_FOLDER, FACTORY_FOLDER

factory_bp = Blueprint("factory_api", __name__, url_prefix="/api/factory")


@factory_bp.route("/trends", methods=["GET"])
def get_trends_endpoint():
    category = request.args.get("category", "all")
    limit = int(request.args.get("limit", 15))
    topics = scout_trending(category=category, limit=limit)
    return jsonify({"topics": [t.to_dict() for t in topics], "count": len(topics)})


@factory_bp.route("/categories", methods=["GET"])
def get_categories_endpoint():
    return jsonify({"categories": get_available_categories()})


@factory_bp.route("/generate-script", methods=["POST"])
def generate_script_endpoint():
    data = request.get_json() or {}
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    target_dur = int(data.get("target_duration", 45))
    tone = data.get("tone", "energetic")
    style = data.get("style", "informative")

    script = generate_script(
        topic=topic,
        topic_description=data.get("description", ""),
        topic_keywords=data.get("keywords", []),
        target_duration=target_dur,
        tone=tone,
        style=style,
    )

    if not script:
        return jsonify({"error": "Failed to generate script via LLM. Check your API keys."}), 500

    return jsonify({"success": True, "script": script.to_dict()})


@factory_bp.route("/generate", methods=["POST"])
def start_factory_generation():
    data = request.get_json() or {}
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    job_id = start_generation(
        topic=topic,
        topic_description=data.get("description", ""),
        topic_keywords=data.get("keywords", []),
        target_duration=int(data.get("target_duration", 45)),
        tone=data.get("tone", "energetic"),
        style=data.get("style", "informative"),
        music_mood=data.get("music_mood", "upbeat"),
        music_volume=float(data.get("music_volume", 0.15)),
        color_grade=data.get("color_grade", "cinematic"),
        video_format=data.get("video_format", "9:16"),
        output_base_dir=FACTORY_FOLDER,
    )

    return jsonify({"job_id": job_id, "status": "pending"})


@factory_bp.route("/custom-generate", methods=["POST"])
def start_custom_factory_generation():
    job_id = str(uuid.uuid4())[:8]
    title = request.form.get("title", "Custom Story").strip()
    script_text = request.form.get("script_text", "").strip()

    if not script_text:
        return jsonify({"error": "Script text is required"}), 400

    image_paths = []
    if "images" in request.files:
        files = request.files.getlist("images")
        job_tmp = os.path.join(UPLOAD_FOLDER, f"custom_{job_id}")
        os.makedirs(job_tmp, exist_ok=True)

        for f in files:
            if f.filename:
                safe_name = secure_filename(f.filename)
                save_p = os.path.join(job_tmp, safe_name)
                f.save(save_p)
                image_paths.append(save_p)

    music_mood = request.form.get("music_mood", "upbeat")
    music_volume = float(request.form.get("music_volume", 0.15))
    video_format = request.form.get("video_format", "9:16")

    start_custom_generation(
        job_id=job_id,
        title=title,
        script_text=script_text,
        image_paths=image_paths,
        music_mood=music_mood,
        music_volume=music_volume,
        video_format=video_format,
        output_base_dir=FACTORY_FOLDER,
    )

    return jsonify({"job_id": job_id, "status": "pending"})


@factory_bp.route("/jobs", methods=["GET"])
def list_factory_jobs_endpoint():
    return jsonify({"jobs": list_jobs()})


@factory_bp.route("/job/<job_id>", methods=["GET"])
def get_factory_job_endpoint(job_id: str):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job.to_dict())


@factory_bp.route("/job/<job_id>", methods=["DELETE"])
def delete_factory_job_endpoint(job_id: str):
    deleted = cleanup_job(job_id)
    if not deleted:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"success": True})


@factory_bp.route("/styles", methods=["GET"])
def get_styles_endpoint():
    return jsonify({"styles": get_style_presets()})


@factory_bp.route("/voices", methods=["GET"])
def get_voices_endpoint():
    return jsonify({"voices": get_available_voices()})


@factory_bp.route("/moods", methods=["GET"])
def get_moods_endpoint():
    return jsonify({"moods": get_mood_options()})


@factory_bp.route("/commentary/generate", methods=["POST"])
def generate_commentary_endpoint():
    """Start an AI commentary voiceover generation job."""
    data = request.get_json() or {}
    video_path = data.get("video_path", "").strip()
    style = data.get("style", "reaction")

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Valid video_path is required"}), 400

    from video_clipper.web.context import set_job, update_job, GLOBAL_THREAD_POOL
    from video_clipper.clipping.transcriber import transcribe
    from video_clipper.content_generation.commentary import generate_commentary

    job_id = str(uuid.uuid4())[:8]
    set_job(job_id, {"status": "transcribing", "progress": 10, "message": "Transcribing audio..."})

    def _bg_commentary(jid: str, vpath: str, st: str):
        try:
            transcript = transcribe(vpath)
            update_job(jid, {"status": "generating", "progress": 50, "message": "Generating AI commentary script..."})
            script = generate_commentary(
                transcript=transcript,
                video_title=Path(vpath).stem,
                video_duration=transcript.duration,
                style=st,
            )
            update_job(jid, {
                "status": "completed",
                "progress": 100,
                "message": "Commentary script generated",
                "script": script.to_dict() if script else None,
            })
        except Exception as e:
            update_job(jid, {"status": "error", "error": str(e)})

    GLOBAL_THREAD_POOL.submit(_bg_commentary, job_id, video_path, style)
    return jsonify({"job_id": job_id, "status": "processing"})


@factory_bp.route("/commentary/status/<job_id>", methods=["GET"])
def get_commentary_status(job_id: str):
    from video_clipper.web.context import get_job_state
    state = get_job_state(job_id)
    if not state:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(state)

