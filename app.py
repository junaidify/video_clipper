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
                        _get_cookies_config, _get_cookies_config_with_fallback,
                        _is_running_locally, _detect_browser)
from llm_analyzer import analyze_with_llm, LLMConfig
from library import VideoLibrary
from manual_clipper import TimestampClip, parse_timestamp, split_by_timestamps
from sequential_clipper import SequentialConfig, split_sequentially
from trainer import PatternTrainer
from subtitle_generator import generate_subtitles, burn_subtitles, burn_text_overlay
from thumbnail_generator import generate_template_thumbnail, generate_ai_thumbnail, pick_best_frame
import youtube_uploader

# ─── App Setup ───
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['CLIPS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'clips_output')
app.config['LIBRARY_FOLDER'] = os.path.join(os.path.dirname(__file__), 'video_library')
app.config['TRAINING_FOLDER'] = os.path.join(os.path.dirname(__file__), 'training_sessions')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'video-clipper-dev-key')

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'm4v'}

# In-memory job tracker (thread-safe via lock)
jobs = {}
jobs_lock = threading.Lock()

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
    """Add a video to the library from URL or file upload.
    URL downloads run as background jobs with progress tracking.
    File uploads remain synchronous (fast enough).
    """
    # ── URL path (async with progress) ──
    url = request.form.get('url', '').strip()
    if url:
        if not is_valid_url(url):
            return jsonify({'error': 'Invalid URL. Please enter a valid HTTP/HTTPS link.'}), 400
        drm = is_drm_platform(url)
        if drm:
            return jsonify({'error': f'{drm} uses DRM protection and cannot be downloaded. This applies to all streaming services like Netflix, Disney+, Hulu, etc.'}), 400

        job_id = str(uuid.uuid4())[:8]
        jobs[job_id] = {
            'status': 'downloading',
            'progress': 5,
            'message': 'Connecting...',
            'error': None,
            'video': None,
            'started_at': time.time(),
        }
        thread = threading.Thread(target=_library_download_job, args=(job_id, url), daemon=True)
        thread.start()
        return jsonify({'success': True, 'job_id': job_id})

    # ── File upload path (synchronous — fast enough) ──
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
            try:
                os.remove(tmp_path)
            except OSError:
                pass

            return jsonify({'success': True, 'video': entry.to_dict()})

    return jsonify({'error': 'No valid video file or URL provided'}), 400


def _library_download_job(job_id: str, url: str):
    """Background job: download video with real-time progress updates."""
    try:
        import yt_dlp

        jobs[job_id]['message'] = 'Resolving video info...'
        jobs[job_id]['progress'] = 10

        # Progress hook — yt-dlp calls this during download
        def _progress_hook(d):
            if d['status'] == 'downloading':
                total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                downloaded = d.get('downloaded_bytes', 0)
                speed = d.get('speed') or 0
                eta = d.get('eta') or 0

                if total > 0:
                    pct = min(85, int(15 + (downloaded / total) * 70))
                    size_mb = total / (1024 * 1024)
                    dl_mb = downloaded / (1024 * 1024)
                    jobs[job_id]['progress'] = pct
                    jobs[job_id]['message'] = f'Downloading: {dl_mb:.1f} / {size_mb:.1f} MB'
                else:
                    dl_mb = downloaded / (1024 * 1024)
                    jobs[job_id]['message'] = f'Downloading: {dl_mb:.1f} MB'

                if speed > 0:
                    speed_mb = speed / (1024 * 1024)
                    jobs[job_id]['message'] += f' ({speed_mb:.1f} MB/s)'
                if eta > 0:
                    jobs[job_id]['message'] += f' — {eta}s left'

            elif d['status'] == 'finished':
                jobs[job_id]['progress'] = 88
                jobs[job_id]['message'] = 'Download complete, processing...'

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        # Use download_video but with progress hook injected
        dl_result = download_video(url, app.config['UPLOAD_FOLDER'],
                                   progress_hook=_progress_hook)

        if not dl_result.success:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = f'Download failed: {dl_result.error}'
            jobs[job_id]['progress'] = 0
            return

        jobs[job_id]['progress'] = 92
        jobs[job_id]['message'] = 'Saving to library...'

        duration = _get_video_duration(dl_result.file_path)
        entry = video_library.add_video(
            source_path=dl_result.file_path,
            title=dl_result.title or Path(dl_result.file_path).stem,
            source='url',
            source_url=url,
            duration=duration,
        )

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['message'] = 'Added to library!'
        jobs[job_id]['video'] = entry.to_dict()

    except Exception as e:
        logger.exception("Library download job failed")
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['progress'] = 0


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
        jobs[job_id]['message'] = f'Loading Whisper model ({settings["model_size"]})...'
        jobs[job_id]['progress'] = 10

        # Simulate sub-steps for transcription (the longest phase)
        def _transcribe_progress_update():
            """Gradually increment progress during transcription."""
            import time as _time
            step = 15
            while jobs[job_id]['status'] == 'transcribing' and step < 48:
                _time.sleep(5)
                if jobs[job_id]['status'] != 'transcribing':
                    break
                step = min(step + 3, 48)
                jobs[job_id]['progress'] = step
                if step < 20:
                    jobs[job_id]['message'] = f'Loading Whisper model ({settings["model_size"]})...'
                elif step < 35:
                    jobs[job_id]['message'] = f'Transcribing audio (model: {settings["model_size"]})...'
                else:
                    jobs[job_id]['message'] = f'Processing transcript segments...'

        progress_thread = threading.Thread(target=_transcribe_progress_update, daemon=True)
        progress_thread.start()

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
        total_clips = len(candidates)
        jobs[job_id]['message'] = f'Splitting clip 1 of {total_clips}...'

        clipper_config = ClipperConfig(
            min_clip_duration=settings['min_duration'],
            max_clip_duration=settings['max_duration'],
            crop_vertical=settings['crop_vertical'],
        )
        clipper = VideoClipper(clipper_config)

        # Split one at a time with per-clip progress
        results = []
        for i, cand in enumerate(candidates):
            jobs[job_id]['message'] = f'Splitting clip {i+1} of {total_clips}...'
            jobs[job_id]['progress'] = 65 + int((i / total_clips) * 30)
            clip_results = clipper.extract_all_clips(
                video_path=video_path,
                candidates=[cand],
                output_dir=output_dir,
                video_duration=transcript.duration,
            )
            results.extend(clip_results)

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
        import tempfile
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


@app.route('/api/cookie-status', methods=['GET'])
def cookie_diagnostic():
    """Diagnostic endpoint to check cookie file health on deployed instances."""
    result = {
        'env_var_set': bool(os.getenv('YOUTUBE_COOKIES_B64')),
        'env_var_length': len(os.getenv('YOUTUBE_COOKIES_B64', '')),
    }

    # Check cookie file at expected path
    cookie_path = os.path.join(os.path.dirname(__file__), 'cookies.txt')
    result['cookie_file_path'] = cookie_path
    result['cookie_file_exists'] = os.path.isfile(cookie_path)

    if result['cookie_file_exists']:
        stat = os.stat(cookie_path)
        result['cookie_file_size'] = stat.st_size
        with open(cookie_path, 'r', errors='replace') as f:
            lines = f.readlines()
        result['cookie_file_lines'] = len(lines)
        # Check format (first line should mention Netscape/HTTP Cookie)
        first_line = lines[0].strip() if lines else ''
        result['first_line_preview'] = first_line[:80]
        result['valid_netscape_format'] = 'netscape' in first_line.lower() or 'http cookie' in first_line.lower()
        # Count actual cookie entries (non-comment, non-empty lines)
        cookie_entries = [l for l in lines if l.strip() and not l.startswith('#')]
        result['cookie_entries'] = len(cookie_entries)
        # Check for youtube.com domain entries
        yt_entries = [l for l in cookie_entries if 'youtube' in l.lower() or 'google' in l.lower()]
        result['youtube_cookie_entries'] = len(yt_entries)
    else:
        result['cookie_file_size'] = 0
        result['reason'] = 'Cookie file not found. Check YOUTUBE_COOKIES_B64 env var and container startup logs.'

    # What _get_cookies_config returns
    cookies_config = _get_cookies_config()
    result['active_config'] = cookies_config if cookies_config else 'none'

    return jsonify(result)


@app.route('/api/upload-cookies', methods=['POST'])
def upload_cookies():
    """Upload a cookies.txt file for YouTube authentication."""
    if 'cookie_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['cookie_file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    content = file.read().decode('utf-8', errors='replace')

    # Basic validation
    if len(content.strip()) < 50:
        return jsonify({'error': 'File is too small to be a valid cookie file'}), 400

    # Check for YouTube/Google cookies
    lines = content.strip().split('\n')
    yt_entries = [l for l in lines if l.strip() and not l.startswith('#')
                  and ('youtube' in l.lower() or 'google' in l.lower())]

    if not yt_entries:
        return jsonify({'error': 'No YouTube/Google cookies found in this file. '
                                 'Make sure you export cookies while on youtube.com.'}), 400

    # Save to project root as cookies.txt
    cookie_path = os.path.join(os.path.dirname(__file__), 'cookies.txt')
    with open(cookie_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info(f"Cookie file uploaded: {len(yt_entries)} YouTube entries, "
                f"{len(lines)} total lines, saved to {cookie_path}")

    return jsonify({
        'success': True,
        'youtube_entries': len(yt_entries),
        'total_lines': len(lines),
    })


# ═══════════════════════════════════════════════════
#  YOUTUBE UPLOAD
# ═══════════════════════════════════════════════════

@app.route('/api/youtube/status')
def youtube_status():
    """Check YouTube connection status."""
    return jsonify({
        'configured': youtube_uploader.is_configured(),
        'authenticated': youtube_uploader.is_authenticated(),
    })


@app.route('/api/youtube/auth')
def youtube_auth():
    """Start OAuth2 flow — redirect user to Google consent screen."""
    if not youtube_uploader.is_configured():
        return jsonify({'error': 'GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET not set in .env'}), 400
    try:
        redirect_uri = request.url_root.rstrip('/') + '/api/youtube/callback'
        auth_url = youtube_uploader.get_auth_url(redirect_uri)
        return jsonify({'auth_url': auth_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/youtube/callback')
def youtube_callback():
    """OAuth2 callback — exchange code for tokens."""
    code = request.args.get('code')
    error = request.args.get('error')
    if error:
        return f"""<html><body><h2>YouTube Auth Failed</h2><p>{error}</p>
        <script>window.close();</script></body></html>"""
    if not code:
        return 'Missing authorization code', 400
    try:
        redirect_uri = request.url_root.rstrip('/') + '/api/youtube/callback'
        youtube_uploader.handle_oauth_callback(code, redirect_uri)
        return """<html><body style="background:#0a0a0a;color:#fff;font-family:system-ui;
        display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
        <div style="text-align:center">
        <h2 style="color:#a3e635">YouTube Connected!</h2>
        <p>You can close this tab and return to CLIPPER.</p>
        <script>
            if(window.opener){window.opener.postMessage({type:'youtube_auth_done'},'*');}
            setTimeout(()=>window.close(),2000);
        </script></div></body></html>"""
    except Exception as e:
        logger.exception("YouTube OAuth callback failed")
        return f"""<html><body><h2>Auth Error</h2><p>{e}</p>
        <script>setTimeout(()=>window.close(),5000);</script></body></html>"""


@app.route('/api/youtube/disconnect', methods=['POST'])
def youtube_disconnect():
    """Remove stored YouTube tokens."""
    youtube_uploader.disconnect()
    return jsonify({'success': True})


@app.route('/api/youtube/generate-metadata', methods=['POST'])
def youtube_generate_metadata():
    """Use NVIDIA NIM (OpenAI-compatible) to generate optimized YouTube metadata for a clip."""
    data = request.get_json() or {}
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')

    if not job_id or not filename:
        return jsonify({'error': 'Missing job_id or filename'}), 400

    clip_dir = _find_job_dir(job_id)
    clip_path = os.path.join(clip_dir, filename)
    if not os.path.isfile(clip_path):
        return jsonify({'error': 'Clip file not found'}), 404

    # Try to get transcript/subtitle text for context
    stem = Path(filename).stem
    srt_path = os.path.join(clip_dir, f"{stem}.srt")
    transcript_text = ''
    if os.path.isfile(srt_path):
        with open(srt_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            text_lines = [l.strip() for l in lines
                         if l.strip() and not l.strip().isdigit()
                         and '-->' not in l]
            transcript_text = ' '.join(text_lines)

    # Get clip duration
    duration = _get_video_duration(clip_path)

    # Check for NVIDIA API key
    llm_config = LLMConfig.from_env()
    if not llm_config.nvidia_api_key:
        # Fallback: generate basic metadata without AI
        meta = youtube_uploader.generate_clip_metadata(
            clip_info={'hook_text': transcript_text[:200], 'duration': duration},
            source_title=filename,
            clip_number=1, total_clips=1,
        )
        return jsonify({
            'title': meta['title'],
            'description': meta['description'],
            'tags': meta['tags'],
            'ai_generated': False,
        })

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=llm_config.nvidia_api_key,
        )

        prompt = f"""You are a YouTube Shorts optimization expert. Generate metadata for a short video clip to MAXIMIZE reach, views, and engagement.

Clip info:
- Filename: {filename}
- Duration: {int(duration)} seconds
- Transcript/content: {transcript_text[:1500] if transcript_text else 'No transcript available. Generate based on filename.'}

Generate the following in JSON format:
{{
    "title": "Catchy, click-worthy title (max 80 chars). Use power words, curiosity gaps, or emotional hooks. DO NOT use generic titles.",
    "description": "Engaging description (max 300 chars). Include context, call to action, and relevant keywords. Add #Shorts at the end.",
    "tags": ["tag1", "tag2", ...] (10-15 highly relevant tags for discoverability. Mix broad and niche tags.)
}}

Rules:
- Title must be scroll-stopping and make people WANT to click
- Tags should include trending/searchable terms related to the content
- Description should feel natural, not spammy
- Output ONLY valid JSON, nothing else"""

        response = client.chat.completions.create(
            model=llm_config.nvidia_model,
            messages=[
                {"role": "system", "content": "You are a viral content analyst and YouTube SEO expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()

        result = json.loads(raw)
        return jsonify({
            'title': result.get('title', filename)[:100],
            'description': result.get('description', '')[:5000],
            'tags': result.get('tags', [])[:15],
            'ai_generated': True,
        })

    except Exception as e:
        logger.exception("NVIDIA metadata generation failed")
        # Fallback to basic metadata
        meta = youtube_uploader.generate_clip_metadata(
            clip_info={'hook_text': transcript_text[:200], 'duration': duration},
            source_title=filename,
            clip_number=1, total_clips=1,
        )
        return jsonify({
            'title': meta['title'],
            'description': meta['description'],
            'tags': meta['tags'],
            'ai_generated': False,
            'warning': f'NVIDIA NIM failed ({str(e)[:80]}), using basic metadata',
        })


@app.route('/api/youtube/upload-single', methods=['POST'])
def youtube_upload_single():
    """Upload a single clip to YouTube with user-approved metadata."""
    data = request.get_json() or {}
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')
    privacy = data.get('privacy', 'unlisted')
    category = data.get('category', 'entertainment')
    custom_title = data.get('title', '').strip()
    custom_description = data.get('description', '').strip()
    custom_tags = data.get('tags', [])
    schedule_at = data.get('schedule_at', '').strip()  # ISO 8601 datetime

    if not job_id or not filename:
        return jsonify({'error': 'Missing job_id or filename'}), 400

    if not youtube_uploader.is_authenticated():
        return jsonify({'error': 'YouTube not connected. Click "Connect YouTube" first.'}), 401

    clip_dir = _find_job_dir(job_id)
    file_path = os.path.join(clip_dir, filename)
    if not os.path.isfile(file_path):
        return jsonify({'error': 'Clip file not found'}), 404

    # Use custom metadata from preview form, fallback to filename
    title = custom_title or Path(filename).stem
    description = custom_description or ''
    tags = custom_tags if isinstance(custom_tags, list) else []

    # Validate schedule_at if provided
    publish_at = None
    if schedule_at:
        from datetime import datetime, timezone
        try:
            # Parse and ensure it's in the future
            dt = datetime.fromisoformat(schedule_at.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            if dt <= now:
                return jsonify({'error': 'Scheduled time must be in the future'}), 400
            # Convert to ISO 8601 with Z suffix for YouTube API
            publish_at = dt.strftime('%Y-%m-%dT%H:%M:%S.0Z')
        except ValueError:
            return jsonify({'error': 'Invalid schedule datetime format'}), 400

    result = youtube_uploader.upload_video(
        file_path=file_path,
        title=title,
        description=description,
        tags=tags,
        category=category,
        privacy=privacy,
        publish_at=publish_at,
    )

    if not result.success:
        return jsonify({'error': result.error}), 500

    # Upload subtitles if SRT exists
    stem = Path(filename).stem
    srt_path = os.path.join(clip_dir, f"{stem}.srt")
    if os.path.isfile(srt_path) and result.video_id:
        youtube_uploader.upload_caption(result.video_id, srt_path)

    resp = {
        'success': True,
        'video_id': result.video_id,
        'video_url': result.video_url,
    }
    if publish_at:
        resp['scheduled_at'] = publish_at
    return jsonify(resp)


@app.route('/api/youtube/upload', methods=['POST'])
def youtube_upload():
    """Batch upload clips from a job to YouTube."""
    data = request.get_json() or {}
    job_id = data.get('job_id', '')
    privacy = data.get('privacy', 'private')
    category = data.get('category', 'entertainment')
    source_title = data.get('source_title', '')
    custom_tags = data.get('tags', [])
    made_for_kids = data.get('made_for_kids', False)
    clips_meta = data.get('clips', [])  # optional per-clip overrides

    if not job_id:
        return jsonify({'error': 'Missing job_id'}), 400

    if not youtube_uploader.is_authenticated():
        return jsonify({'error': 'YouTube not connected. Click "Connect YouTube" first.'}), 401

    # Find clip files
    clip_dir = _find_job_dir(job_id)
    if not os.path.isdir(clip_dir):
        return jsonify({'error': 'Clip directory not found'}), 404

    clip_files = sorted([
        f for f in os.listdir(clip_dir)
        if f.lower().endswith(('.mp4', '.webm', '.mov'))
        and '_subtitled' not in f.lower()
        and '_overlay' not in f.lower()
        and '_thumb_' not in f.lower()
    ])

    if not clip_files:
        return jsonify({'error': 'No clips found in this job'}), 404

    # Start background upload job
    upload_job_id = f"yt_{job_id}_{str(uuid.uuid4())[:4]}"
    jobs[upload_job_id] = {
        'status': 'uploading',
        'progress': 0,
        'message': f'Starting upload of {len(clip_files)} clips...',
        'total': len(clip_files),
        'uploaded': 0,
        'results': [],
        'error': None,
    }

    thread = threading.Thread(
        target=_youtube_upload_job,
        args=(upload_job_id, clip_dir, clip_files, privacy, category,
              source_title, custom_tags, made_for_kids, clips_meta),
        daemon=True,
    )
    thread.start()
    return jsonify({'upload_job_id': upload_job_id, 'total': len(clip_files)})


def _youtube_upload_job(upload_job_id, clip_dir, clip_files, privacy, category,
                        source_title, custom_tags, made_for_kids, clips_meta):
    """Background: upload all clips to YouTube one by one."""
    total = len(clip_files)
    results = []

    for i, filename in enumerate(clip_files):
        file_path = os.path.join(clip_dir, filename)
        jobs[upload_job_id]['current_file'] = filename
        jobs[upload_job_id]['message'] = f'Uploading {i+1}/{total}: {filename}'

        # Get metadata — either from clips_meta or auto-generate
        clip_info = {}
        if clips_meta and i < len(clips_meta):
            clip_info = clips_meta[i]

        meta = youtube_uploader.generate_clip_metadata(
            clip_info=clip_info,
            source_title=source_title,
            clip_number=i + 1,
            total_clips=total,
        )

        # Allow custom tag override
        if custom_tags:
            meta['tags'] = list(set(meta['tags'] + custom_tags))[:15]

        # Use custom title if provided in clip_info
        title = clip_info.get('custom_title') or meta['title']
        description = clip_info.get('custom_description') or meta['description']

        def _progress(sent, total_bytes):
            if total_bytes > 0:
                file_pct = int((sent / total_bytes) * 100)
                overall_pct = int(((i + file_pct / 100) / total) * 100)
                jobs[upload_job_id]['progress'] = overall_pct
                jobs[upload_job_id]['message'] = (
                    f'Uploading {i+1}/{total}: {filename} ({file_pct}%)'
                )

        result = youtube_uploader.upload_video(
            file_path=file_path,
            title=title,
            description=description,
            tags=meta['tags'],
            category=category,
            privacy=privacy,
            made_for_kids=made_for_kids,
            progress_callback=_progress,
        )

        # Try uploading subtitles if SRT exists
        if result.success and result.video_id:
            stem = Path(filename).stem
            srt_path = os.path.join(clip_dir, f"{stem}.srt")
            if os.path.isfile(srt_path):
                youtube_uploader.upload_caption(result.video_id, srt_path)
                logger.info(f"Captions uploaded for {filename}")

        results.append({
            'filename': result.filename,
            'success': result.success,
            'video_id': result.video_id,
            'video_url': result.video_url,
            'error': result.error,
        })

        jobs[upload_job_id]['uploaded'] = i + 1
        jobs[upload_job_id]['results'] = results

        # If upload failed due to quota, stop
        if result.error and 'quota' in result.error.lower():
            jobs[upload_job_id]['status'] = 'error'
            jobs[upload_job_id]['error'] = (
                f'YouTube daily quota exceeded after {i+1} uploads. '
                f'Try again tomorrow or request quota increase in Google Cloud Console.'
            )
            jobs[upload_job_id]['message'] = 'Quota exceeded'
            return

    success_count = sum(1 for r in results if r['success'])
    fail_count = total - success_count

    jobs[upload_job_id]['status'] = 'completed'
    jobs[upload_job_id]['progress'] = 100

    if fail_count == 0:
        jobs[upload_job_id]['message'] = f'All {success_count} clips uploaded!'
    else:
        jobs[upload_job_id]['message'] = f'{success_count} uploaded, {fail_count} failed'
        if success_count == 0:
            jobs[upload_job_id]['status'] = 'error'
            jobs[upload_job_id]['error'] = results[0].get('error', 'All uploads failed')


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
