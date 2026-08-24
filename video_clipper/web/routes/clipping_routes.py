"""
Clipping Routes Blueprint
API endpoints for Smart AI Clipping, Manual Timestamp Splitting,
Sequential Reel Cutting, Medium-Length Engagement Highlights, and Full Video Processing.
"""
import os
import uuid
from pathlib import Path
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from video_clipper.config import PipelineConfig, TranscriberConfig, AnalyzerConfig, ClipperConfig
from video_clipper.clipping.manual_splitter import TimestampClip, parse_timestamp, split_by_timestamps
from video_clipper.clipping.sequential_splitter import SequentialConfig, split_sequentially
from video_clipper.clipping.engagement_analyzer import (
    EngagementConfig, EngagementAnalyzer, analyze_with_llm_for_engagement
)
from video_clipper.clipping.clipper import VideoClipper
from video_clipper.clipping.transcriber import transcribe
from video_clipper.clipping.full_video_processor import FullVideoConfig
from video_clipper.web.context import (
    get_job_state,
    set_job,
    get_full_video_job_state,
    get_video_library,
    UPLOAD_FOLDER,
    OUTPUT_FOLDER,
)
from video_clipper.web.job_manager import (
    start_smart_clip_job,
    start_full_video_job,
)

clipping_bp = Blueprint("clipping_api", __name__, url_prefix="/api")


@clipping_bp.route("/clip", methods=["POST"])
def start_smart_clipping():
    """Start an asynchronous Smart AI Clipping job."""
    job_id = str(uuid.uuid4())[:8]

    source_type = request.form.get("source_type", "upload")
    video_source_value = None
    custom_title = None

    if source_type == "library":
        video_id = request.form.get("video_id", "").strip()
        if not video_id:
            return jsonify({"error": "video_id is required for library source"}), 400
        video_source_value = video_id
    elif source_type == "url":
        url = request.form.get("url", "").strip()
        if not url:
            return jsonify({"error": "url is required for url source"}), 400
        video_source_value = url
        custom_title = request.form.get("title", "").strip() or None
    else:
        # File upload
        if "video" not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400
        file = request.files["video"]
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400

        safe_name = f"{job_id}_{secure_filename(file.filename)}"
        save_path = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(save_path)
        video_source_value = save_path
        custom_title = request.form.get("title", "").strip() or Path(file.filename).stem

    # Build PipelineConfig from form fields
    model_size = request.form.get("whisper_model", "base")
    language = request.form.get("language") or None
    min_score = float(request.form.get("min_hook_score", 0.4))
    max_clips = int(request.form.get("max_clips", 10))
    min_dur = int(request.form.get("min_clip_duration", 15))
    max_dur = int(request.form.get("max_clip_duration", 60))
    crop_vertical = request.form.get("crop_vertical", "true").lower() == "true"
    fade_dur = float(request.form.get("fade_duration", 0.5))

    pipeline_cfg = PipelineConfig(
        transcriber=TranscriberConfig(model_size=model_size, language=language),
        analyzer=AnalyzerConfig(min_hook_score=min_score, max_clips=max_clips),
        clipper=ClipperConfig(
            min_clip_duration=min_dur,
            max_clip_duration=max_dur,
            crop_vertical=crop_vertical,
            fade_duration=fade_dur,
        ),
    )

    # Anti-copyright settings
    anti_copyright = request.form.get("anti_copyright", "false").lower() == "true"
    anti_copyright_config = {
        "zoom_percent": float(request.form.get("zoom_percent", 3.0)),
        "mirror_flip": request.form.get("mirror_flip", "false").lower() == "true",
        "color_grade": request.form.get("color_grade", "warm"),
        "grain_overlay": request.form.get("grain_overlay", "true").lower() == "true",
        "grain_intensity": float(request.form.get("grain_intensity", 0.04)),
        "speed_shift": float(request.form.get("speed_shift", 1.0)),
    }

    session_id = request.form.get("session_id") or None

    start_smart_clip_job(
        job_id=job_id,
        video_source_type=source_type,
        video_source_value=video_source_value,
        pipeline_config=pipeline_cfg,
        anti_copyright=anti_copyright,
        anti_copyright_config=anti_copyright_config,
        session_id=session_id,
        title=custom_title,
    )

    return jsonify({"job_id": job_id, "status": "running", "message": "Smart clip job started."})


@clipping_bp.route("/status/<job_id>", methods=["GET"])
def get_clipping_status(job_id: str):
    """Query progress and result of a smart clipping job."""
    state = get_job_state(job_id)
    if not state:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(state)


@clipping_bp.route("/manual-split", methods=["POST"])
def manual_split_video():
    """Split video at exact user-defined timestamps."""
    data = request.get_json() or {}
    video_path = data.get("video_path", "").strip()
    raw_clips = data.get("clips", [])
    crop_vertical = data.get("crop_vertical", True)

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Valid video_path is required"}), 400

    if not raw_clips:
        return jsonify({"error": "At least one clip interval is required"}), 400

    clip_objects = []
    for c in raw_clips:
        try:
            start = parse_timestamp(str(c.get("start", 0)))
            end = parse_timestamp(str(c.get("end", 0)))
            label = c.get("label")
            clip_objects.append(TimestampClip(start=start, end=end, label=label))
        except Exception as e:
            return jsonify({"error": f"Invalid timestamp: {e}"}), 400

    stem = Path(video_path).stem
    out_dir = os.path.join(OUTPUT_FOLDER, f"manual_{stem}_{str(uuid.uuid4())[:6]}")

    results = split_by_timestamps(video_path, clip_objects, out_dir, crop_vertical=crop_vertical)
    return jsonify({"success": True, "results": results, "output_dir": out_dir})


@clipping_bp.route("/sequential-split", methods=["POST"])
def sequential_split_video():
    """Split entire video into consecutive reels with overlap."""
    data = request.get_json() or {}
    video_path = data.get("video_path", "").strip()

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Valid video_path is required"}), 400

    target_dur = int(data.get("target_duration", 55))
    overlap = float(data.get("overlap_seconds", 1.5))
    crop_vert = data.get("crop_vertical", True)

    config = SequentialConfig(
        target_duration=target_dur,
        overlap_seconds=overlap,
        crop_vertical=crop_vert,
    )

    stem = Path(video_path).stem
    out_dir = os.path.join(OUTPUT_FOLDER, f"sequential_{stem}_{str(uuid.uuid4())[:6]}")

    reels = split_sequentially(video_path, out_dir, config)
    return jsonify({"success": True, "reels": reels, "output_dir": out_dir, "count": len(reels)})


@clipping_bp.route("/engagement-clips", methods=["POST"])
def extract_engagement_clips():
    """Analyze and extract medium-length (5-20 min) highlights."""
    data = request.get_json() or {}
    video_path = data.get("video_path", "").strip()

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Valid video_path is required"}), 400

    min_dur = int(data.get("min_segment_duration", 300))
    max_dur = int(data.get("max_segment_duration", 1200))
    max_segs = int(data.get("max_segments", 5))

    cfg = EngagementConfig(
        min_segment_duration=min_dur,
        max_segment_duration=max_dur,
        max_segments=max_segs,
    )

    # Transcribe
    stem = Path(video_path).stem
    out_dir = os.path.join(OUTPUT_FOLDER, f"engagement_{stem}_{str(uuid.uuid4())[:6]}")
    os.makedirs(out_dir, exist_ok=True)

    transcript = transcribe(video_path, cache_dir=out_dir)

    analyzer = EngagementAnalyzer(cfg)
    segments = analyzer.analyze(transcript)

    # Cut clips
    clipper = VideoClipper()
    results = []
    for i, seg in enumerate(segments, 1):
        out_name = f"engagement_{i:02d}_{seg.start:.0f}s-{seg.end:.0f}s.mp4"
        out_path = os.path.join(out_dir, out_name)
        success = clipper.cut_clip(video_path, seg.start, seg.end, out_path)
        results.append({
            "segment_number": i,
            "title_hint": seg.title_hint,
            "summary": seg.summary,
            "start": seg.start,
            "end": seg.end,
            "duration": seg.duration,
            "score": seg.score,
            "reason": seg.reason,
            "output_path": out_path,
            "success": success,
        })

    return jsonify({"success": True, "segments": results, "output_dir": out_dir})


@clipping_bp.route("/process-full-video", methods=["POST"])
def process_full_length_video():
    """Start asynchronous full video enhancement processing."""
    data = request.get_json() or {}
    video_path = data.get("video_path", "").strip()

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Valid video_path is required"}), 400

    job_id = str(uuid.uuid4())[:8]
    stem = Path(video_path).stem
    out_dir = os.path.join(OUTPUT_FOLDER, f"full_{stem}_{job_id}")

    config = FullVideoConfig(
        crop_vertical=data.get("crop_vertical", True),
        enable_subtitles=data.get("enable_subtitles", False),
        subtitle_model_size=data.get("subtitle_model_size", "base"),
        subtitle_language=data.get("subtitle_language"),
        enable_overlay=data.get("enable_overlay", False),
        overlay_text=data.get("overlay_text", ""),
        enable_thumbnail=data.get("enable_thumbnail", False),
        thumbnail_title=data.get("thumbnail_title", ""),
        enable_modulation=data.get("enable_modulation", False),
        modulation_preset=data.get("modulation_preset", ""),
    )

    start_full_video_job(job_id, video_path, out_dir, config)
    return jsonify({"job_id": job_id, "status": "running"})


@clipping_bp.route("/full-video-status/<job_id>", methods=["GET"])
def get_full_video_status(job_id: str):
    state = get_full_video_job_state(job_id)
    if not state:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(state)


@clipping_bp.route("/cancel/<job_id>", methods=["POST"])
def cancel_clipping_job(job_id: str):
    set_job(job_id, status="cancelled", message="Job cancelled by user.")
    return jsonify({"success": True})
