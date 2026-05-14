"""
Flask Web Application for Video Auto-Clipper
Full panel UI: Library, Smart Clips, Manual Split, Sequential Reels, Training.
"""
import json
import logging
import os
import subprocess
import sys
import time
import uuid
import threading
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig, TranscriberConfig, AnalyzerConfig, ClipperConfig
from transcriber import transcribe
from analyzer import ContentAnalyzer, ClipCandidate
from clipper import VideoClipper
from downloader import (is_valid_url, is_drm_platform, download_video,
                        get_video_info, get_supported_platforms,
                        _get_cookies_config, _is_running_locally, _detect_browser)
from llm_analyzer import analyze_with_llm, LLMConfig
from library import VideoLibrary
from manual_clipper import TimestampClip, parse_timestamp, split_by_timestamps
from sequential_clipper import SequentialConfig, split_sequentially
from trainer import PatternTrainer
from subtitle_generator import generate_subtitles, burn_subtitles, burn_text_overlay
from thumbnail_generator import generate_template_thumbnail, generate_ai_thumbnail, pick_best_frame

# ─── App Setup ───
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['CLIPS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'clips_output')
app.config['LIBRARY_FOLDER'] = os.path.join(os.path.dirname(__file__), 'video_library')
app.config['TRAINING_FOLDER'] = os.path.join(os.path.dirname(__file__), 'training_sessions')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'video-clipper-dev-key')

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'm4v'}

# In-memory job tracker
jobs = {}

# Persistent modules
video_library = VideoLibrary(app.config['LIBRARY_FOLDER'])
trainer = PatternTrainer(app.config['TRAINING_FOLDER'])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


import re
import datetime

def _sanitize_folder_name(title: str) -> str:
    """Create filesystem-safe folder name from video title."""
    safe = re.sub(r'[^\w\s-]', '', title)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:50] if len(safe) > 50 else (safe or "video")


def _make_clip_output_dir(video_title: str, job_id: str) -> str:
    """
    Build output dir: clips_output/{VideoTitle_timestamp}/{job_id}/
    The parent folder groups all sessions for the same source video.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_name = f"{_sanitize_folder_name(video_title)}_{ts}"
    # Check if a folder for this video already exists (match by title prefix)
    clips_root = app.config['CLIPS_FOLDER']
    prefix = _sanitize_folder_name(video_title)
    existing_parent = None
    if os.path.isdir(clips_root):
        for d in os.listdir(clips_root):
            if d.startswith(prefix) and os.path.isdir(os.path.join(clips_root, d)):
                existing_parent = d
                break
    parent = existing_parent or parent_name
    job_dir = os.path.join(clips_root, parent, job_id)
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


def _get_video_duration(video_path: str) -> float:
    """Get video duration via ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", video_path
        ]
        r = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        info = json.loads(r.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════
#  PAGE ROUTES
# ═══════════════════════════════════════════════════

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/app')
def app_ui():
    return render_template('app.html')


@app.route('/editor')
def editor_page():
    """Dedicated clip editor page for post-processing a single clip."""
    return render_template('editor.html')


# ═══════════════════════════════════════════════════
#  VIDEO LIBRARY ROUTES
# ═══════════════════════════════════════════════════

@app.route('/api/library/list')
def library_list():
    """List all videos in the library."""
    videos = video_library.list_videos()
    return jsonify({
        'videos': [v.to_dict() for v in videos],
        'stats': video_library.get_library_stats(),
    })


@app.route('/api/library/add', methods=['POST'])
def library_add():
    """Add a video to the library from URL or file upload."""
    # ── URL path ──
    url = request.form.get('url', '').strip()
    if url:
        if not is_valid_url(url):
            return jsonify({'error': 'Invalid URL. Please enter a valid HTTP/HTTPS link.'}), 400
        drm = is_drm_platform(url)
        if drm:
            return jsonify({'error': f'{drm} uses DRM protection and cannot be downloaded. This applies to all streaming services like Netflix, Disney+, Hulu, etc.'}), 400
        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            dl_result = download_video(url, app.config['UPLOAD_FOLDER'])
            if not dl_result.success:
                return jsonify({'error': f'Download failed: {dl_result.error}'}), 400

            duration = _get_video_duration(dl_result.file_path)
            entry = video_library.add_video(
                source_path=dl_result.file_path,
                title=dl_result.title or Path(dl_result.file_path).stem,
                source='url',
                source_url=url,
                duration=duration,
            )
            return jsonify({'success': True, 'video': entry.to_dict()})
        except Exception as e:
            logger.exception("Library add from URL failed")
            return jsonify({'error': str(e)}), 500

    # ── File upload path ──
    if 'video' in request.files:
        file = request.files['video']
        if file and file.filename and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = secure_filename(file.filename)
            tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"lib_{uuid.uuid4().hex[:8]}_{filename}")
            file.save(tmp_path)

            duration = _get_video_duration(tmp_path)
            title = Path(filename).stem
            entry = video_library.add_video(
                source_path=tmp_path,
                title=title,
                source='upload',
                duration=duration,
            )
            # Clean up temp upload (library made its own copy)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

            return jsonify({'success': True, 'video': entry.to_dict()})

    return jsonify({'error': 'No valid video file or URL provided'}), 400


@app.route('/api/library/delete', methods=['POST'])
def library_delete():
    """Delete a video from the library."""
    data = request.get_json() or {}
    video_id = data.get('video_id', '')
    if not video_id:
        return jsonify({'error': 'Missing video_id'}), 400
    ok = video_library.delete_video(video_id)
    return jsonify({'success': ok})


# ═══════════════════════════════════════════════════
#  SMART CLIPS (AI-analyzed hook detection)
# ═══════════════════════════════════════════════════

@app.route('/api/smart-clip', methods=['POST'])
def smart_clip():
    """Start smart clip processing using a library video."""
    video_id = request.form.get('video_id', '')
    entry = video_library.get_video(video_id) if video_id else None
    if not entry:
        return jsonify({'error': 'Select a video from the library'}), 400

    if not Path(entry.file_path).exists():
        return jsonify({'error': 'Video file missing from library'}), 400

    settings = {
        'model_size': request.form.get('model_size', 'base'),
        'max_clips': int(request.form.get('max_clips', 10)),
        'min_score': float(request.form.get('min_score', 0.4)),
        'min_duration': int(request.form.get('min_duration', 15)),
        'max_duration': int(request.form.get('max_duration', 60)),
        'crop_vertical': request.form.get('crop_vertical', 'true') == 'true',
        'use_llm': request.form.get('use_llm', 'false') == 'true',
    }

    job_id = str(uuid.uuid4())[:8]
    job_dir = _make_clip_output_dir(entry.title, job_id)

    jobs[job_id] = {
        'status': 'transcribing',
        'progress': 5,
        'message': 'Starting transcription...',
        'clips': [],
        'error': None,
    }

    thread = threading.Thread(
        target=_smart_clip_job,
        args=(job_id, entry.file_path, video_id, job_dir, settings),
        daemon=True,
    )
    thread.start()
    return jsonify({'job_id': job_id, 'status': 'started'})


def _smart_clip_job(job_id, video_path, video_id, output_dir, settings):
    """Background: transcribe → analyze → split."""
    try:
        # Transcribe
        jobs[job_id]['status'] = 'transcribing'
        jobs[job_id]['message'] = f'Transcribing (model: {settings["model_size"]})...'
        jobs[job_id]['progress'] = 20

        transcript = transcribe(
            video_path=video_path,
            model_size=settings['model_size'],
        )
        jobs[job_id]['message'] = f'Transcribed: {len(transcript.segments)} segments'
        jobs[job_id]['progress'] = 50
        transcript.save(os.path.join(output_dir, 'transcript.json'))

        # Analyze
        jobs[job_id]['status'] = 'analyzing'
        jobs[job_id]['message'] = 'Analyzing for hooks...'
        jobs[job_id]['progress'] = 55

        analyzer_config = AnalyzerConfig(
            min_hook_score=settings['min_score'],
            max_clips=settings['max_clips'],
        )
        analyzer = ContentAnalyzer(analyzer_config)
        candidates = analyzer.analyze(transcript)

        # LLM fallback
        if settings.get('use_llm') and len(candidates) < 2:
            jobs[job_id]['message'] = 'Trying LLM analysis...'
            llm_candidates = analyze_with_llm(transcript)
            if llm_candidates:
                for lc in llm_candidates:
                    candidates.append(ClipCandidate(
                        start=lc['start'], end=lc['end'],
                        score=lc['score'], hook_text=lc['hook_text'],
                        reason=f"llm_{lc['reason']}",
                    ))
                candidates.sort(key=lambda c: c.score, reverse=True)
                candidates = candidates[:settings['max_clips']]

        if not candidates:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['message'] = 'No hooks found. Lower the min score.'
            jobs[job_id]['progress'] = 100
            return

        jobs[job_id]['message'] = f'{len(candidates)} candidates found'
        jobs[job_id]['progress'] = 65

        # Split
        jobs[job_id]['status'] = 'splitting'
        jobs[job_id]['message'] = 'Splitting clips...'

        clipper_config = ClipperConfig(
            min_clip_duration=settings['min_duration'],
            max_clip_duration=settings['max_duration'],
            crop_vertical=settings['crop_vertical'],
        )
        clipper = VideoClipper(clipper_config)
        results = clipper.extract_all_clips(
            video_path=video_path,
            candidates=candidates,
            output_dir=output_dir,
            video_duration=transcript.duration,
        )

        clips_info = []
        for r in results:
            if r.success:
                fname = Path(r.output_path).name
                clips_info.append({
                    'clip_number': r.clip_number,
                    'filename': fname,
                    'url': f'/clips/{job_id}/{fname}',
                    'start': r.start, 'end': r.end,
                    'duration': r.duration,
                    'score': r.score,
                    'reason': r.reason,
                    'hook_text': r.hook_text[:200],
                })

        # Record output dir in library
        video_library.add_clips_directory(video_id, output_dir)

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['clips'] = clips_info
        jobs[job_id]['message'] = f'Done! {len(clips_info)} clips.'
        jobs[job_id]['progress'] = 100

    except Exception as e:
        logger.exception(f"Smart clip job {job_id} failed")
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['message'] = f'Error: {e}'


# ═══════════════════════════════════════════════════
#  MANUAL TIMESTAMP SPLIT
# ═══════════════════════════════════════════════════

@app.route('/api/manual-split', methods=['POST'])
def manual_split():
    """Split video at user-defined timestamps."""
    video_id = request.form.get('video_id', '')
    entry = video_library.get_video(video_id) if video_id else None
    if not entry:
        return jsonify({'error': 'Select a video from the library'}), 400
    if not Path(entry.file_path).exists():
        return jsonify({'error': 'Video file missing'}), 400

    raw_clips = json.loads(request.form.get('clips', '[]'))
    if not raw_clips:
        return jsonify({'error': 'No timestamps provided'}), 400

    crop = request.form.get('crop_vertical', 'true') == 'true'

    # Parse timestamps
    try:
        clips = []
        for c in raw_clips:
            start = parse_timestamp(c['start'])
            end = parse_timestamp(c['end'])
            clips.append(TimestampClip(start=start, end=end, label=c.get('label')))
    except (ValueError, KeyError) as e:
        return jsonify({'error': f'Invalid timestamp: {e}'}), 400

    job_id = str(uuid.uuid4())[:8]
    job_dir = _make_clip_output_dir(entry.title, job_id)

    results = split_by_timestamps(
        video_path=entry.file_path,
        clips=clips,
        output_dir=job_dir,
        crop_vertical=crop,
    )

    # Record in library
    video_library.add_clips_directory(video_id, job_dir)

    return jsonify({
        'job_id': job_id,
        'results': results,
        'success_count': sum(1 for r in results if r.get('success')),
    })


# ═══════════════════════════════════════════════════
#  SEQUENTIAL REELS (full video → consecutive shorts)
# ═══════════════════════════════════════════════════

@app.route('/api/sequential-split', methods=['POST'])
def sequential_split():
    """Split entire video into consecutive reels."""
    video_id = request.form.get('video_id', '')
    entry = video_library.get_video(video_id) if video_id else None
    if not entry:
        return jsonify({'error': 'Select a video from the library'}), 400
    if not Path(entry.file_path).exists():
        return jsonify({'error': 'Video file missing'}), 400

    target_dur = int(request.form.get('target_duration', 55))
    overlap = float(request.form.get('overlap', 1.5))
    crop = request.form.get('crop_vertical', 'true') == 'true'
    use_transcript = request.form.get('use_transcript', 'false') == 'true'

    job_id = str(uuid.uuid4())[:8]
    job_dir = _make_clip_output_dir(entry.title, job_id)

    jobs[job_id] = {
        'status': 'processing',
        'progress': 10,
        'message': 'Starting sequential split...',
        'reels': [],
        'error': None,
    }

    thread = threading.Thread(
        target=_sequential_job,
        args=(job_id, entry.file_path, video_id, job_dir,
              target_dur, overlap, crop, use_transcript),
        daemon=True,
    )
    thread.start()
    return jsonify({'job_id': job_id, 'status': 'started'})


def _sequential_job(job_id, video_path, video_id, output_dir,
                    target_dur, overlap, crop, use_transcript):
    """Background: sequential split."""
    try:
        config = SequentialConfig(
            target_duration=target_dur,
            overlap_seconds=overlap,
            crop_vertical=crop,
        )

        transcript = None
        if use_transcript:
            jobs[job_id]['message'] = 'Transcribing for sentence boundaries...'
            jobs[job_id]['progress'] = 20
            transcript = transcribe(video_path=video_path, model_size='base')

        jobs[job_id]['message'] = 'Splitting into reels...'
        jobs[job_id]['progress'] = 40

        duration = _get_video_duration(video_path)
        results = split_sequentially(
            video_path=video_path,
            output_dir=output_dir,
            config=config,
            transcript=transcript,
            video_duration=duration,
        )

        success_reels = [r for r in results if r.get('success')]
        failed_reels = [r for r in results if not r.get('success')]
        video_library.add_clips_directory(video_id, output_dir)

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['reels'] = success_reels
        jobs[job_id]['progress'] = 100

        if success_reels:
            jobs[job_id]['message'] = f'Done! {len(success_reels)} reels.'
        elif failed_reels:
            first_err = failed_reels[0].get('error', 'Unknown')
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = f'All {len(failed_reels)} reels failed. First error: {first_err}'
            jobs[job_id]['message'] = f'FFmpeg failed on all reels'
        else:
            jobs[job_id]['message'] = 'No split points computed (video too short?)'

        logger.info(f"Sequential job {job_id}: {len(success_reels)} ok, {len(failed_reels)} failed")

    except Exception as e:
        logger.exception(f"Sequential job {job_id} failed")
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['message'] = f'Error: {e}'


# ═══════════════════════════════════════════════════
#  TRAINING (pattern extraction from examples)
# ═══════════════════════════════════════════════════

@app.route('/api/training/create', methods=['POST'])
def training_create():
    """Create a new private training session."""
    session_id = trainer.create_session()
    return jsonify({'session_id': session_id})


@app.route('/api/training/sessions')
def training_sessions():
    """List all training sessions."""
    return jsonify({'sessions': trainer.list_sessions()})


@app.route('/api/training/upload', methods=['POST'])
def training_upload():
    """Upload a video to a training session (long or short form)."""
    session_id = request.form.get('session_id', '')
    vtype = request.form.get('type', '')  # 'long' or 'short'

    if not session_id:
        return jsonify({'error': 'No session selected'}), 400
    if vtype not in ('long', 'short'):
        return jsonify({'error': 'Type must be "long" or "short"'}), 400

    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['video']
    if not file or not file.filename:
        return jsonify({'error': 'Empty file'}), 400

    # Save temp file
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename = secure_filename(file.filename)
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"train_{uuid.uuid4().hex[:8]}_{filename}")
    file.save(tmp_path)

    try:
        # Get duration
        duration = _get_video_duration(tmp_path)

        # Quick transcription for pattern analysis
        transcript = transcribe(video_path=tmp_path, model_size='tiny')

        transcript_data = {
            'duration': duration,
            'segments': [{'start': s.start, 'end': s.end, 'text': s.text}
                         for s in transcript.segments],
            'full_text': transcript.full_text if hasattr(transcript, 'full_text')
                         else ' '.join(s.text for s in transcript.segments),
        }

        if vtype == 'long':
            trainer.add_long_form(session_id, transcript_data)
        else:
            trainer.add_short_form(session_id, transcript_data)

        return jsonify({'success': True, 'duration': duration})

    except Exception as e:
        logger.exception("Training upload failed")
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.route('/api/training/extract', methods=['POST'])
def training_extract():
    """Extract patterns from a training session."""
    data = request.get_json() or {}
    session_id = data.get('session_id', '')
    if not session_id:
        return jsonify({'error': 'No session_id'}), 400

    try:
        profile = trainer.extract_patterns(session_id)
        return jsonify({'success': True, 'profile': profile.to_dict()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/delete', methods=['POST'])
def training_delete():
    """Delete a training session."""
    data = request.get_json() or {}
    session_id = data.get('session_id', '')
    ok = trainer.delete_session(session_id) if session_id else False
    return jsonify({'success': ok})


# ═══════════════════════════════════════════════════
#  SHARED ROUTES
# ═══════════════════════════════════════════════════

@app.route('/api/status/<job_id>')
def job_status(job_id):
    """Get processing status for a job."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(jobs[job_id])


def _find_job_dir(job_id: str) -> str:
    """Find the actual directory for a job_id, searching nested structure."""
    clips_root = app.config['CLIPS_FOLDER']
    # Direct path (legacy: clips_output/job_id/)
    direct = os.path.join(clips_root, job_id)
    if os.path.isdir(direct):
        return direct
    # Nested path (new: clips_output/parent_folder/job_id/)
    if os.path.isdir(clips_root):
        for parent in os.listdir(clips_root):
            nested = os.path.join(clips_root, parent, job_id)
            if os.path.isdir(nested):
                return nested
    return direct  # fallback


@app.route('/clips/<job_id>/<filename>')
def serve_clip(job_id, filename):
    """Serve a generated clip file."""
    clip_dir = _find_job_dir(job_id)
    return send_from_directory(clip_dir, filename)


@app.route('/api/clips/list/<job_id>')
def list_clips_in_dir(job_id):
    """List all clip files in a job/clips directory."""
    clip_dir = _find_job_dir(job_id)
    if not os.path.isdir(clip_dir):
        return jsonify({'files': [], 'error': 'Directory not found'})
    files = []
    for f in sorted(os.listdir(clip_dir)):
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            fpath = os.path.join(clip_dir, f)
            size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
            files.append({
                'filename': f,
                'url': f'/clips/{job_id}/{f}',
                'size_mb': size_mb,
            })
    return jsonify({'files': files, 'job_id': job_id})


# ═══════════════════════════════════════════════════
#  SUBTITLE, OVERLAY, THUMBNAIL ROUTES
# ═══════════════════════════════════════════════════

@app.route('/api/clips/subtitle', methods=['POST'])
def add_subtitles():
    """Generate and burn subtitles into a clip."""
    data = request.get_json()
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')
    model_size = data.get('model_size', 'base')

    if not job_id or not filename:
        return jsonify({'error': 'job_id and filename required'}), 400

    clip_path = os.path.join(_find_job_dir(job_id), filename)
    if not os.path.isfile(clip_path):
        return jsonify({'error': 'Clip file not found'}), 404

    try:
        # Generate subtitles
        result = generate_subtitles(clip_path, model_size=model_size)

        # Build output filename
        stem = Path(filename).stem
        ext = Path(filename).suffix
        out_name = f"{stem}_subtitled{ext}"
        out_path = os.path.join(_find_job_dir(job_id), out_name)

        # Burn into video
        burn_subtitles(clip_path, result['srt'], out_path)

        # Save SRT file alongside
        srt_name = f"{stem}.srt"
        srt_path = os.path.join(_find_job_dir(job_id), srt_name)
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(result['srt'])

        return jsonify({
            'success': True,
            'method': result['method'],
            'subtitled_url': f'/clips/{job_id}/{out_name}',
            'srt_url': f'/clips/{job_id}/{srt_name}',
            'word_count': len(result['words']),
        })
    except Exception as e:
        logger.exception("Subtitle generation failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/clips/overlay', methods=['POST'])
def add_text_overlay():
    """Burn text overlay onto a clip at specific positions with per-word colors."""
    data = request.get_json()
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')
    text_blocks = data.get('text_blocks', [])

    if not job_id or not filename:
        return jsonify({'error': 'job_id and filename required'}), 400
    if not text_blocks:
        return jsonify({'error': 'text_blocks required'}), 400

    clip_path = os.path.join(_find_job_dir(job_id), filename)
    if not os.path.isfile(clip_path):
        return jsonify({'error': 'Clip file not found'}), 404

    try:
        stem = Path(filename).stem
        ext = Path(filename).suffix
        out_name = f"{stem}_overlay{ext}"
        out_path = os.path.join(_find_job_dir(job_id), out_name)

        burn_text_overlay(clip_path, out_path, text_blocks)

        return jsonify({
            'success': True,
            'overlay_url': f'/clips/{job_id}/{out_name}',
        })
    except Exception as e:
        logger.exception("Text overlay failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/clips/frame', methods=['POST'])
def get_clip_frame():
    """Extract a single frame from a clip for the overlay editor preview."""
    data = request.get_json()
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')
    timestamp = data.get('timestamp', 0.5)  # seconds into clip

    if not job_id or not filename:
        return jsonify({'error': 'job_id and filename required'}), 400

    clip_path = os.path.join(_find_job_dir(job_id), filename)
    if not os.path.isfile(clip_path):
        return jsonify({'error': 'Clip file not found'}), 404

    try:
        import tempfile, base64
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            frame_path = tmp.name

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", clip_path,
            "-vframes", "1",
            "-q:v", "2",
            frame_path
        ]
        subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', check=True)

        with open(frame_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        os.unlink(frame_path)

        # Get video dimensions
        cmd2 = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", clip_path
        ]
        result = subprocess.run(cmd2, capture_output=True, text=True, encoding='utf-8', errors='replace')
        dims = json.loads(result.stdout)["streams"][0]

        return jsonify({
            'success': True,
            'frame': f'data:image/png;base64,{img_b64}',
            'width': dims['width'],
            'height': dims['height'],
        })
    except Exception as e:
        logger.exception("Frame extraction failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/clips/frames', methods=['POST'])
def get_clip_frames():
    """Extract multiple frames from a clip for the overlay editor frame picker."""
    data = request.get_json()
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')
    num_frames = min(data.get('num_frames', 8), 16)

    if not job_id or not filename:
        return jsonify({'error': 'job_id and filename required'}), 400

    clip_path = os.path.join(_find_job_dir(job_id), filename)
    if not os.path.isfile(clip_path):
        return jsonify({'error': 'Clip file not found'}), 404

    try:
        import base64 as b64mod
        # Get duration
        probe_cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", clip_path
        ]
        probe_res = subprocess.run(probe_cmd, capture_output=True, text=True,
                                   encoding='utf-8', errors='replace')
        probe_data = json.loads(probe_res.stdout)
        duration = float(probe_data["format"]["duration"])
        dims = probe_data["streams"][0]
        w, h = int(dims["width"]), int(dims["height"])

        frames = []
        interval = duration / (num_frames + 1)
        for i in range(1, num_frames + 1):
            ts = interval * i
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                frame_path = tmp.name
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(ts),
                "-i", clip_path,
                "-vframes", "1",
                "-vf", "scale=320:-1",
                "-q:v", "4",
                frame_path
            ]
            result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
            if result.returncode == 0 and os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                with open(frame_path, "rb") as f:
                    img_b64 = b64mod.b64encode(f.read()).decode()
                frames.append({
                    'timestamp': round(ts, 2),
                    'thumb': f'data:image/jpeg;base64,{img_b64}',
                })
            if os.path.exists(frame_path):
                os.unlink(frame_path)

        return jsonify({
            'success': True,
            'frames': frames,
            'width': w,
            'height': h,
            'duration': duration,
        })
    except Exception as e:
        logger.exception("Multi-frame extraction failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/clips/thumbnail', methods=['POST'])
def generate_thumbnail():
    """Generate a thumbnail for a clip. Supports 'template' and 'ai' modes."""
    data = request.get_json()
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')
    title = data.get('title', 'Untitled')
    mode = data.get('mode', 'template')  # 'template' or 'ai'
    style = data.get('style', 'bold')

    if not job_id or not filename:
        return jsonify({'error': 'job_id and filename required'}), 400

    clip_path = os.path.join(_find_job_dir(job_id), filename)
    if not os.path.isfile(clip_path):
        return jsonify({'error': 'Clip file not found'}), 404

    try:
        stem = Path(filename).stem
        thumb_name = f"{stem}_thumb_{mode}.png"
        thumb_path = os.path.join(_find_job_dir(job_id), thumb_name)

        if mode == 'ai':
            generate_ai_thumbnail(clip_path, title, thumb_path, context="")
        else:
            generate_template_thumbnail(clip_path, title, thumb_path, style=style)

        return jsonify({
            'success': True,
            'thumbnail_url': f'/clips/{job_id}/{thumb_name}',
            'mode': mode,
        })
    except Exception as e:
        logger.exception(f"Thumbnail generation ({mode}) failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/check-url', methods=['POST'])
def check_url():
    """Check if a URL is valid and get video info."""
    data = request.get_json()
    url = data.get('url', '').strip()
    if not url or not is_valid_url(url):
        return jsonify({'valid': False, 'error': 'Invalid URL. Enter a valid HTTP/HTTPS link.'})
    drm = is_drm_platform(url)
    if drm:
        return jsonify({'valid': False, 'error': f'{drm} uses DRM protection and cannot be downloaded.'})
    try:
        info = get_video_info(url)
        return jsonify({'valid': True, 'info': info})
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)})


@app.route('/api/platforms', methods=['GET'])
def list_platforms():
    """Return supported platform info for the UI."""
    return jsonify(get_supported_platforms())


@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Return API key status and dependency checks."""
    import shutil
    config = LLMConfig.from_env()
    cookies_config = _get_cookies_config()
    if 'cookiesfrombrowser' in cookies_config:
        cookie_status = f"auto ({cookies_config['cookiesfrombrowser']})"
    elif 'cookiefile' in cookies_config:
        cookie_status = "cookie file"
    else:
        cookie_status = "none"

    return jsonify({
        'groq_configured': bool(config.groq_api_key),
        'gemini_configured': bool(config.gemini_api_key),
        'nvidia_configured': bool(config.nvidia_api_key),
        'llm_available': config.has_any_key(),
        'ffmpeg_installed': shutil.which('ffmpeg') is not None,
        'is_local': _is_running_locally(),
        'cookie_auth': cookie_status,
        'detected_browser': _detect_browser(),
    })


# ═══════════════════════════════════════════════════
#  LEGACY /api/process (kept for backward compat)
# ═══════════════════════════════════════════════════

@app.route('/api/process', methods=['POST'])
def process_video():
    """Legacy: Start processing from direct upload/URL (not library)."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(app.config['CLIPS_FOLDER'], job_id)
    os.makedirs(job_dir, exist_ok=True)

    settings = {
        'model_size': request.form.get('model_size', 'base'),
        'max_clips': int(request.form.get('max_clips', 10)),
        'min_score': float(request.form.get('min_score', 0.4)),
        'min_duration': int(request.form.get('min_duration', 15)),
        'max_duration': int(request.form.get('max_duration', 60)),
        'crop_vertical': request.form.get('crop_vertical', 'true') == 'true',
        'use_llm': request.form.get('use_llm', 'false') == 'true',
    }

    # Legacy route redirects users to the new library-based flow
    return jsonify({
        'error': 'This endpoint is deprecated. Use the Video Library panel to add videos, '
                 'then use Smart Clips / Manual Split / Sequential Reels.'
    }), 400


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CLIPS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['LIBRARY_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TRAINING_FOLDER'], exist_ok=True)

    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'

    print(f"""
+==================================================+
|          VIDEO AUTO-CLIPPER  (Web UI)             |
|   Library + Smart + Manual + Sequential + Train   |
|   http://localhost:{port}                          |
+==================================================+
    """)

    if debug:
        # Dev mode: Flask's built-in server with hot reload
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        # Production: Waitress (cross-platform, Windows + Linux)
        try:
            from waitress import serve
            print(f"  [Waitress] Serving on http://0.0.0.0:{port}")
            serve(app, host='0.0.0.0', port=port, threads=8,
                  channel_timeout=300, recv_bytes=65536)
        except ImportError:
            logger.warning("Waitress not installed, falling back to Flask dev server")
            app.run(host='0.0.0.0', port=port, debug=False)
