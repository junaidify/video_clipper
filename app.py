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
from core.transcriber import transcribe
from core.analyzer import ContentAnalyzer, ClipCandidate
from core.clipper import VideoClipper
from media.downloader import (is_valid_url, is_drm_platform, download_video,
                        get_video_info, get_supported_platforms,
                        _get_cookies_config, _get_cookies_config_with_fallback,
                        _is_running_locally, _detect_browser)
from core.llm_analyzer import analyze_with_llm, LLMConfig
from media.library import VideoLibrary
from modes.manual_clipper import TimestampClip, parse_timestamp, split_by_timestamps
from modes.sequential_clipper import SequentialConfig, split_sequentially
from training.trainer import PatternTrainer
from post.subtitle_generator import generate_subtitles, burn_subtitles, burn_text_overlay
from post.thumbnail_generator import generate_template_thumbnail, generate_ai_thumbnail, pick_best_frame
import publish.youtube_uploader as youtube_uploader
from publish.content_factory import (
    start_generation as factory_start, get_job as factory_get_job,
    list_jobs as factory_list_jobs, cleanup_job as factory_cleanup_job
)
from ai.trend_scout import scout_trending, get_available_categories as get_trend_categories
from ai.script_generator import get_style_presets as get_script_styles
from media.music_mixer import get_mood_options
from modes.full_video import FullVideoConfig, process_full_video
from modes.engagement_clipper import (
    EngagementConfig, EngagementAnalyzer, analyze_with_llm_for_engagement
)

# ─── App Setup ───
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB max upload
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


# Global error handlers — ensure API routes always return JSON, not HTML
@app.errorhandler(404)
def not_found_error(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found', 'path': request.path}), 404
    return e

@app.errorhandler(500)
def internal_error(e):
    if request.path.startswith('/api/'):
        logger.exception("Internal server error on %s", request.path)
        return jsonify({'error': 'Internal server error'}), 500
    return e

@app.errorhandler(405)
def method_not_allowed(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Method not allowed'}), 405
    return e


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

    # ── File upload path ──
    if 'video' in request.files:
        file = request.files['video']
        if file and file.filename and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = secure_filename(file.filename)
            tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"lib_{uuid.uuid4().hex[:8]}_{filename}")
            file.save(tmp_path)

            file_size = os.path.getsize(tmp_path) if os.path.isfile(tmp_path) else 0

            # Large files (>50MB): process async to avoid browser timeout
            if file_size > 50 * 1024 * 1024:
                job_id = str(uuid.uuid4())[:8]
                jobs[job_id] = {
                    'status': 'processing',
                    'progress': 60,
                    'message': f'Processing upload ({file_size / (1024*1024):.0f} MB)...',
                    'error': None,
                    'video': None,
                    'started_at': time.time(),
                }
                thread = threading.Thread(
                    target=_library_upload_job,
                    args=(job_id, tmp_path, filename),
                    daemon=True,
                )
                thread.start()
                return jsonify({'success': True, 'job_id': job_id})

            # Small files: synchronous (fast)
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


def _library_upload_job(job_id: str, tmp_path: str, filename: str):
    """Background job: process large uploaded file into library."""
    try:
        jobs[job_id]['message'] = 'Analyzing video...'
        jobs[job_id]['progress'] = 70

        duration = _get_video_duration(tmp_path)

        jobs[job_id]['message'] = 'Saving to library...'
        jobs[job_id]['progress'] = 85

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

        jobs[job_id]['video'] = entry.to_dict()
        jobs[job_id]['progress'] = 100
        jobs[job_id]['message'] = 'Added to library!'
        jobs[job_id]['status'] = 'completed'

    except Exception as e:
        logger.exception("Upload processing failed")
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['progress'] = 0


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
            uploader=dl_result.uploader,
            channel_url=dl_result.channel_url,
        )

        jobs[job_id]['video'] = entry.to_dict()
        jobs[job_id]['progress'] = 100
        jobs[job_id]['message'] = 'Added to library!'
        jobs[job_id]['status'] = 'completed'  # MUST be last — triggers frontend render

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


@app.route('/api/clips/delete', methods=['POST'])
def clips_delete():
    """Delete a clip file or entire clip folder."""
    import shutil
    data = request.get_json() or {}
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')  # optional — if empty, delete whole folder

    if not job_id:
        return jsonify({'error': 'Missing job_id'}), 400

    clip_dir = _find_job_dir(job_id)
    if not os.path.isdir(clip_dir):
        return jsonify({'error': 'Clip directory not found'}), 404

    if filename:
        # Delete single clip file
        fpath = os.path.join(clip_dir, filename)
        if not os.path.isfile(fpath):
            return jsonify({'error': 'File not found'}), 404
        os.remove(fpath)
        # If folder is now empty of video files, remove the folder too
        remaining = [f for f in os.listdir(clip_dir)
                     if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
        if not remaining:
            shutil.rmtree(clip_dir, ignore_errors=True)
        return jsonify({'success': True, 'deleted': 'file'})
    else:
        # Delete entire clip folder
        shutil.rmtree(clip_dir, ignore_errors=True)
        return jsonify({'success': True, 'deleted': 'folder'})


# ── Upload Status Tracking ──
_upload_status_file = os.path.join(os.path.dirname(__file__), 'upload_status.json')
_upload_status_lock = threading.Lock()


def _load_upload_status():
    with _upload_status_lock:
        if os.path.exists(_upload_status_file):
            try:
                with open(_upload_status_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}


def _save_upload_status(data):
    with _upload_status_lock:
        with open(_upload_status_file, 'w') as f:
            json.dump(data, f, indent=2)


@app.route('/api/upload-status', methods=['GET'])
def get_upload_status():
    """Get all upload status markers."""
    return jsonify(_load_upload_status())


@app.route('/api/upload-status', methods=['POST'])
def set_upload_status():
    """Toggle upload status for a video or clip."""
    data = request.get_json() or {}
    key = data.get('key', '')  # e.g. "video:abc123" or "clip:jobid:filename"
    uploaded = data.get('uploaded', False)
    if not key:
        return jsonify({'error': 'Missing key'}), 400
    status = _load_upload_status()
    if uploaded:
        status[key] = True
    else:
        status.pop(key, None)
    _save_upload_status(status)
    return jsonify({'success': True})


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
        'anti_copyright': True,  # always apply anti-copyright transforms
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
            jobs[job_id]['message'] = 'No hooks found. Lower the min score.'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['status'] = 'completed'  # MUST be last
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
            anti_copyright=settings.get('anti_copyright', True),
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
        failed_clips = []
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
            else:
                failed_clips.append(f"Clip {r.clip_number}: {r.error or 'Unknown error'}")
                logger.error(f"Clip {r.clip_number} failed: {r.error}")

        # Record output dir in library
        video_library.add_clips_directory(video_id, output_dir)

        # Build status message with failure details if any
        if clips_info:
            msg = f'Done! {len(clips_info)} clips.'
            if failed_clips:
                msg += f' ({len(failed_clips)} failed)'
        elif failed_clips:
            # ALL clips failed — surface the first error so user knows why
            msg = f'All {len(failed_clips)} clips failed to extract. {failed_clips[0]}'
        else:
            msg = 'No clips produced.'

        # CRITICAL: Set clips and message BEFORE status to avoid race condition
        # (frontend polls status; if it reads 'completed' before clips is set,
        #  it sees clips=[] and skips rendering)
        jobs[job_id]['clips'] = clips_info
        jobs[job_id]['message'] = msg
        jobs[job_id]['progress'] = 100
        jobs[job_id]['error'] = '\n'.join(failed_clips) if failed_clips and not clips_info else None
        jobs[job_id]['status'] = 'completed'  # MUST be last — triggers frontend render

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

        jobs[job_id]['reels'] = success_reels
        jobs[job_id]['progress'] = 100

        if success_reels:
            jobs[job_id]['message'] = f'Done! {len(success_reels)} reels.'
        elif failed_reels:
            first_err = failed_reels[0].get('error', 'Unknown')
            jobs[job_id]['error'] = f'All {len(failed_reels)} reels failed. First error: {first_err}'
            jobs[job_id]['message'] = f'FFmpeg failed on all reels'
            jobs[job_id]['status'] = 'error'  # set error status last
            logger.info(f"Sequential job {job_id}: {len(success_reels)} ok, {len(failed_reels)} failed")
            return
        else:
            jobs[job_id]['message'] = 'No split points computed (video too short?)'

        jobs[job_id]['status'] = 'completed'  # MUST be last — triggers frontend render

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


def _get_credit_line(job_id: str = '', video_id: str = '') -> str:
    """
    Look up the source video's uploader info and return a credit line.
    Searches by job_id (for clips) or video_id (for library videos).
    Returns empty string if no uploader info found.
    """
    # Try by video_id first (library video direct upload)
    if video_id:
        entry = video_library.get_video(video_id)
        if entry and entry.uploader:
            if entry.channel_url:
                return f"\n\nOriginal by: {entry.uploader}\n{entry.channel_url}"
            return f"\n\nOriginal by: {entry.uploader}"

    # Try by job_id — match clip directory against library entries' clips_directories
    if job_id:
        clip_dir = _find_job_dir(job_id)
        for v in video_library.list_videos():
            if v.uploader and any(clip_dir.startswith(cd) or cd.startswith(os.path.dirname(clip_dir))
                                  for cd in v.clips_directories):
                if v.channel_url:
                    return f"\n\nOriginal by: {v.uploader}\n{v.channel_url}"
                return f"\n\nOriginal by: {v.uploader}"

    return ''


@app.route('/clips/<job_id>/<filename>')
def serve_clip(job_id, filename):
    """Serve a generated clip file."""
    clip_dir = _find_job_dir(job_id)
    return send_from_directory(clip_dir, filename)


@app.route('/library-file/<filename>')
def serve_library_file(filename):
    """Serve a file from the video library folder (e.g. narrated videos)."""
    lib_folder = app.config['LIBRARY_FOLDER']
    return send_from_directory(lib_folder, filename)


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
    data = request.get_json() or {}
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')
    model_size = data.get('model_size', 'base')

    if not job_id or not filename:
        return jsonify({'error': 'job_id and filename required'}), 400

    clip_path = os.path.join(_find_job_dir(job_id), filename)
    if not os.path.isfile(clip_path):
        return jsonify({'error': 'Clip file not found'}), 404

    language = data.get('language', None)  # "en", "hi", or None for auto

    try:
        # Generate subtitles
        result = generate_subtitles(clip_path, model_size=model_size, language=language)

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


# ─── Generate Final Video (preview-then-generate flow) ───
_generate_jobs = {}
_generate_lock = threading.Lock()


@app.route('/api/clips/generate-final', methods=['POST'])
def generate_final_clip():
    """
    Generate a final video from a clip by applying selected enhancements
    in one pass: subtitles + text overlay + thumbnail.
    User previews each individually, then clicks Generate to bake all into one output.
    """
    data = request.get_json() or {}
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')

    # Enhancement options (all optional)
    enable_subtitles = data.get('enable_subtitles', False)
    subtitle_model = data.get('subtitle_model', 'base')
    subtitle_language = data.get('subtitle_language', None)
    enable_overlay = data.get('enable_overlay', False)
    overlay_text_blocks = data.get('overlay_text_blocks', [])
    enable_thumbnail = data.get('enable_thumbnail', False)
    thumbnail_title = data.get('thumbnail_title', '')
    thumbnail_style = data.get('thumbnail_style', 'bold')

    if not job_id or not filename:
        return jsonify({'error': 'job_id and filename required'}), 400

    clip_path = os.path.join(_find_job_dir(job_id), filename)
    if not os.path.isfile(clip_path):
        return jsonify({'error': 'Clip file not found'}), 404

    gen_id = str(uuid.uuid4())[:8]
    gen_job = {
        'id': gen_id,
        'status': 'starting',
        'progress': 0,
        'stage': 'Initializing...',
        'results': {},
        'error': None,
    }
    with _generate_lock:
        _generate_jobs[gen_id] = gen_job

    thread = threading.Thread(
        target=_run_generate_final,
        args=(gen_id, job_id, filename, clip_path,
              enable_subtitles, subtitle_model, subtitle_language,
              enable_overlay, overlay_text_blocks,
              enable_thumbnail, thumbnail_title, thumbnail_style),
        daemon=True
    )
    thread.start()

    return jsonify({'generate_id': gen_id})


@app.route('/api/clips/generate-final/<gen_id>', methods=['GET'])
def generate_final_status(gen_id):
    """Poll generate-final job status."""
    with _generate_lock:
        job = _generate_jobs.get(gen_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


def _run_generate_final(gen_id, job_id, filename, clip_path,
                        enable_subtitles, subtitle_model, subtitle_language,
                        enable_overlay, overlay_text_blocks,
                        enable_thumbnail, thumbnail_title, thumbnail_style):
    """Background thread: apply all selected enhancements into one final video."""
    def _update(**kwargs):
        with _generate_lock:
            _generate_jobs[gen_id].update(kwargs)

    stem = Path(filename).stem
    ext = Path(filename).suffix
    job_dir = _find_job_dir(job_id)
    results = {}
    working_path = clip_path
    steps_applied = []

    try:
        total_steps = sum([enable_subtitles, enable_overlay, enable_thumbnail])
        if total_steps == 0:
            _update(status='done', progress=100, stage='Nothing to generate',
                    results={'info': 'No enhancements selected'})
            return

        step_pct = 90 // max(total_steps, 1)
        current_pct = 5

        # ── Subtitles ──
        if enable_subtitles:
            _update(status='processing', progress=current_pct, stage='Generating subtitles...')
            try:
                sub_result = generate_subtitles(working_path, model_size=subtitle_model,
                                                language=subtitle_language)
                srt_name = f"{stem}.srt"
                srt_path = os.path.join(job_dir, srt_name)
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write(sub_result['srt'])

                sub_out_name = f"{stem}_final{ext}"
                sub_out_path = os.path.join(job_dir, sub_out_name)
                burn_subtitles(working_path, sub_result['srt'], sub_out_path)

                results['subtitles'] = {
                    'srt_url': f'/clips/{job_id}/{srt_name}',
                    'word_count': len(sub_result.get('words', [])),
                }
                working_path = sub_out_path
                steps_applied.append('subtitles')
            except Exception as e:
                logger.warning(f"Generate-final subtitle stage failed: {e}")
                results['subtitles'] = {'error': str(e)}

            current_pct += step_pct

        # ── Text Overlay ──
        if enable_overlay and overlay_text_blocks:
            _update(progress=current_pct, stage='Burning text overlay...')
            try:
                overlay_out_name = f"{stem}_final_overlay{ext}"
                overlay_out_path = os.path.join(job_dir, overlay_out_name)
                burn_text_overlay(working_path, overlay_out_path, overlay_text_blocks)

                results['overlay'] = {'overlay_url': f'/clips/{job_id}/{overlay_out_name}'}
                working_path = overlay_out_path
                steps_applied.append('text_overlay')
            except Exception as e:
                logger.warning(f"Generate-final overlay stage failed: {e}")
                results['overlay'] = {'error': str(e)}

            current_pct += step_pct

        # ── Thumbnail ──
        if enable_thumbnail:
            _update(progress=current_pct, stage='Generating thumbnail...')
            try:
                title = thumbnail_title or stem
                thumb_name = f"{stem}_final_thumb.png"
                thumb_path = os.path.join(job_dir, thumb_name)
                generate_template_thumbnail(working_path, title, thumb_path,
                                            style=thumbnail_style)
                results['thumbnail'] = {'thumbnail_url': f'/clips/{job_id}/{thumb_name}'}
                steps_applied.append('thumbnail')
            except Exception as e:
                logger.warning(f"Generate-final thumbnail stage failed: {e}")
                results['thumbnail'] = {'error': str(e)}

        final_video_name = os.path.basename(working_path)
        _update(
            status='done', progress=100, stage='Complete',
            results=results,
            final_video_url=f'/clips/{job_id}/{final_video_name}',
            steps_applied=steps_applied,
        )

    except Exception as e:
        logger.exception("Generate-final pipeline failed")
        _update(status='error', error=str(e), results=results)


# ─── Auto-Enhance (one-click: subtitles + overlay + thumbnail) ───
_enhance_jobs = {}
_enhance_lock = threading.Lock()


@app.route('/api/clips/suggest-headline', methods=['POST'])
def suggest_headline():
    """Use LLM to suggest a short hook headline for a clip."""
    data = request.get_json() or {}
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')
    hook_text = data.get('hook_text', '')

    if not job_id or not filename:
        return jsonify({'error': 'job_id and filename required'}), 400

    # If hook_text provided from clip metadata, use it; otherwise try transcript
    if not hook_text:
        clip_path = os.path.join(_find_job_dir(job_id), filename)
        if os.path.isfile(clip_path):
            try:
                result = generate_subtitles(clip_path, model_size='base')
                hook_text = ' '.join(w['word'] for w in result.get('words', []))[:300]
            except Exception:
                hook_text = ''

    if not hook_text:
        return jsonify({'headline': 'WATCH THIS', 'source': 'fallback'})

    # Try LLM to generate a punchy 3-5 word headline
    prompt = f"Generate ONE short punchy video headline (3-5 words, ALL CAPS, no quotes, no hashtags) for this clip transcript:\n\n{hook_text[:500]}\n\nHeadline:"
    try:
        llm_cfg = LLMConfig()
        headline = analyze_with_llm(prompt, llm_cfg)
        # Clean up: take first line, strip quotes
        headline = headline.strip().split('\n')[0].strip('"\'').upper()
        if len(headline) > 50:
            headline = headline[:50]
        if not headline:
            headline = 'WATCH THIS'
        return jsonify({'headline': headline, 'source': 'llm'})
    except Exception as e:
        logger.warning(f"LLM headline suggestion failed: {e}")
        # Fallback: take first few words of transcript
        words = hook_text.split()[:5]
        return jsonify({'headline': ' '.join(words).upper(), 'source': 'transcript'})


@app.route('/api/clips/auto-enhance', methods=['POST'])
def auto_enhance_clip():
    """Start async auto-enhance: subtitles + text overlay + thumbnail."""
    data = request.get_json() or {}
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')
    headline = data.get('headline', 'WATCH THIS')

    if not job_id or not filename:
        return jsonify({'error': 'job_id and filename required'}), 400

    clip_path = os.path.join(_find_job_dir(job_id), filename)
    if not os.path.isfile(clip_path):
        return jsonify({'error': 'Clip file not found'}), 404

    enhance_id = str(uuid.uuid4())[:8]
    enhance_job = {
        'id': enhance_id,
        'status': 'starting',
        'progress': 0,
        'stage': 'Initializing...',
        'results': {},
        'error': None,
    }
    with _enhance_lock:
        _enhance_jobs[enhance_id] = enhance_job

    thread = threading.Thread(
        target=_run_auto_enhance,
        args=(enhance_id, job_id, filename, headline, clip_path),
        daemon=True
    )
    thread.start()

    return jsonify({'enhance_id': enhance_id})


@app.route('/api/clips/auto-enhance/<enhance_id>', methods=['GET'])
def auto_enhance_status(enhance_id):
    """Poll auto-enhance job status."""
    with _enhance_lock:
        job = _enhance_jobs.get(enhance_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


def _run_auto_enhance(enhance_id, job_id, filename, headline, clip_path):
    """Background thread: subtitle → overlay → thumbnail pipeline."""
    def _update(**kwargs):
        with _enhance_lock:
            _enhance_jobs[enhance_id].update(kwargs)

    stem = Path(filename).stem
    ext = Path(filename).suffix
    job_dir = _find_job_dir(job_id)
    results = {}

    try:
        # ── Stage 1: Subtitles (0-50%) ──
        _update(status='processing', progress=10, stage='Generating subtitles...')
        try:
            sub_result = generate_subtitles(clip_path, model_size='base')
            srt_name = f"{stem}.srt"
            srt_path = os.path.join(job_dir, srt_name)
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(sub_result['srt'])

            sub_out_name = f"{stem}_enhanced{ext}"
            sub_out_path = os.path.join(job_dir, sub_out_name)
            burn_subtitles(clip_path, sub_result['srt'], sub_out_path)

            results['subtitles'] = {
                'subtitled_url': f'/clips/{job_id}/{sub_out_name}',
                'srt_url': f'/clips/{job_id}/{srt_name}',
                'word_count': len(sub_result.get('words', [])),
            }
            # Use subtitled video as input for next stage
            working_path = sub_out_path
        except Exception as e:
            logger.warning(f"Auto-enhance subtitle stage failed: {e}")
            results['subtitles'] = {'error': str(e)}
            working_path = clip_path  # Continue with original

        # ── Stage 2: Text Overlay (50-80%) ──
        _update(progress=50, stage='Burning text overlay...')
        try:
            overlay_out_name = f"{stem}_enhanced_overlay{ext}"
            overlay_out_path = os.path.join(job_dir, overlay_out_name)

            # Build text block: headline centered at top-third
            text_blocks = [{
                'words': [
                    {'text': w, 'color': '#CAFF00' if i == 0 else '#FFFFFF'}
                    for i, w in enumerate(headline.split())
                ],
                'x': 0.5,
                'y': 0.15,
                'font_size': 48,
                'bg_opacity': 0.6,
            }]
            burn_text_overlay(working_path, overlay_out_path, text_blocks)

            results['overlay'] = {
                'overlay_url': f'/clips/{job_id}/{overlay_out_name}',
            }
            # Final video is the overlay version
            final_video = overlay_out_path
            final_video_name = overlay_out_name
        except Exception as e:
            logger.warning(f"Auto-enhance overlay stage failed: {e}")
            results['overlay'] = {'error': str(e)}
            final_video = working_path
            final_video_name = os.path.basename(working_path)

        # ── Stage 3: Thumbnail (80-100%) ──
        _update(progress=80, stage='Generating thumbnail...')
        try:
            thumb_name = f"{stem}_thumb_auto.png"
            thumb_path = os.path.join(job_dir, thumb_name)
            generate_template_thumbnail(final_video, headline, thumb_path, style='bold')

            results['thumbnail'] = {
                'thumbnail_url': f'/clips/{job_id}/{thumb_name}',
            }
        except Exception as e:
            logger.warning(f"Auto-enhance thumbnail stage failed: {e}")
            results['thumbnail'] = {'error': str(e)}

        _update(
            status='done', progress=100, stage='Complete',
            results=results,
            final_video_url=f'/clips/{job_id}/{final_video_name}',
        )

    except Exception as e:
        logger.exception("Auto-enhance pipeline failed")
        _update(status='error', error=str(e), results=results)


@app.route('/api/clips/modulate', methods=['POST'])
def modulate_clip():
    """Apply pixel modulation transforms to a clip for hash uniqueness."""
    from post.video_modulator import modulate_video, ModulationConfig, get_presets

    data = request.get_json() or {}
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')
    preset = data.get('preset', '')  # preset name OR empty for custom
    custom_config = data.get('config', {})  # custom config if no preset

    if not job_id or not filename:
        return jsonify({'error': 'job_id and filename required'}), 400

    clip_dir = _find_job_dir(job_id)
    clip_path = os.path.join(clip_dir, filename)
    if not os.path.isfile(clip_path):
        return jsonify({'error': 'Clip file not found'}), 404

    try:
        # Build config from preset or custom values
        if preset:
            presets = get_presets()
            if preset not in presets:
                return jsonify({'error': f'Unknown preset: {preset}. Options: {list(presets.keys())}'}), 400
            cfg_dict = presets[preset]["config"]
            config = ModulationConfig(**cfg_dict)
        else:
            config = ModulationConfig(
                zoom_percent=float(custom_config.get('zoom_percent', 4.0)),
                mirror_flip=bool(custom_config.get('mirror_flip', False)),
                color_grade=custom_config.get('color_grade', 'warm'),
                grain_overlay=bool(custom_config.get('grain_overlay', True)),
                grain_intensity=float(custom_config.get('grain_intensity', 0.05)),
                saturation_boost=float(custom_config.get('saturation_boost', 1.10)),
                black_crush=float(custom_config.get('black_crush', 0.05)),
                speed_shift=float(custom_config.get('speed_shift', 1.0)),
                burn_subtitles=bool(custom_config.get('burn_subtitles', False)),
            )

        # Check for subtitle file if burn_subtitles requested
        if config.burn_subtitles:
            stem = Path(filename).stem
            srt_path = os.path.join(clip_dir, f"{stem}.srt")
            if os.path.isfile(srt_path):
                config.subtitle_path = srt_path

        # Output path: original_name_mod.mp4
        stem = Path(filename).stem
        ext = Path(filename).suffix
        output_name = f"{stem}_mod{ext}"
        output_path = os.path.join(clip_dir, output_name)

        result = modulate_video(clip_path, output_path, config)

        if not result.success:
            return jsonify({'error': result.error}), 500

        return jsonify({
            'success': True,
            'modulated_url': f'/clips/{job_id}/{output_name}',
            'modulated_filename': output_name,
            'transforms': result.transforms_applied,
            'original_size_mb': round(result.original_size / 1024 / 1024, 2),
            'modulated_size_mb': round(result.modulated_size / 1024 / 1024, 2),
            'duration': round(result.duration, 1),
        })

    except Exception as e:
        logger.exception("Video modulation failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/clips/modulation-presets', methods=['GET'])
def get_modulation_presets():
    """Return available modulation presets for the UI."""
    from post.video_modulator import get_presets
    return jsonify(get_presets())


@app.route('/api/clips/top-thumbnails', methods=['POST'])
def get_top_thumbnails():
    """Score and return the top N visually striking frames for thumbnail selection."""
    from post.thumbnail_generator import pick_top_frames

    data = request.get_json() or {}
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')
    top_n = int(data.get('top_n', 5))

    if not job_id or not filename:
        return jsonify({'error': 'job_id and filename required'}), 400

    clip_dir = _find_job_dir(job_id)
    clip_path = os.path.join(clip_dir, filename)
    if not os.path.isfile(clip_path):
        return jsonify({'error': 'Clip file not found'}), 404

    try:
        top_frames = pick_top_frames(clip_path, num_candidates=20, top_n=top_n)

        # Save scored frames to clip dir and build URLs
        results = []
        for i, frame in enumerate(top_frames):
            stem = Path(filename).stem
            thumb_name = f"{stem}_thumb_candidate_{i}.png"
            thumb_dest = os.path.join(clip_dir, thumb_name)

            # Move frame from temp to clip dir
            import shutil
            shutil.move(frame["path"], thumb_dest)

            results.append({
                "url": f"/clips/{job_id}/{thumb_name}",
                "filename": thumb_name,
                "timestamp": round(frame["timestamp"], 2),
                "score": frame["score"],
                "rank": i + 1,
            })

        return jsonify({
            'success': True,
            'thumbnails': results,
            'total_analyzed': 20,
        })

    except Exception as e:
        logger.exception("Thumbnail scoring failed")
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
#  AI COMMENTARY (Narration / Voiceover)
# ═══════════════════════════════════════════════════

@app.route('/api/commentary/voices')
def commentary_voices():
    """List available TTS voices."""
    from media.tts_engine import get_available_voices
    return jsonify({'voices': get_available_voices()})


@app.route('/api/commentary/generate', methods=['POST'])
def commentary_generate():
    """
    Generate commentary script + synthesize TTS + mix with video.
    Full pipeline in one background job.
    """
    data = request.get_json() or {}
    job_id = data.get('job_id', '')
    filename = data.get('filename', '')
    video_id = data.get('video_id', '')  # for library videos
    voice = data.get('voice', 'guy_narrator')
    style = data.get('style', 'narration')
    mode = data.get('mode', 'replace')          # 'replace' or 'mix'
    original_volume = float(data.get('original_volume', 0.2))
    rate = data.get('rate', '+0%')
    pitch = data.get('pitch', '+0Hz')
    custom_prompt = data.get('custom_prompt', '')
    whisper_model = data.get('whisper_model', 'base')

    # Resolve video path
    video_path = None
    if job_id and filename:
        clip_dir = _find_job_dir(job_id)
        candidate = os.path.join(clip_dir, filename)
        if os.path.isfile(candidate):
            video_path = candidate
    elif video_id:
        entry = video_library.get_video(video_id)
        if entry and os.path.isfile(entry.file_path):
            video_path = entry.file_path

    if not video_path:
        return jsonify({'error': 'Video file not found'}), 400

    commentary_job_id = str(uuid.uuid4())[:8]
    jobs[commentary_job_id] = {
        'status': 'starting',
        'progress': 0,
        'message': 'Initializing commentary pipeline...',
        'result': None,
        'error': None,
    }

    thread = threading.Thread(
        target=_commentary_job,
        args=(commentary_job_id, video_path, voice, style, mode,
              original_volume, rate, pitch, custom_prompt, whisper_model),
        daemon=True
    )
    thread.start()

    return jsonify({'job_id': commentary_job_id})


@app.route('/api/commentary/status/<cjob_id>')
def commentary_status(cjob_id):
    """Poll commentary job progress."""
    job = jobs.get(cjob_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


def _commentary_job(cjob_id, video_path, voice, style, mode,
                    original_volume, rate, pitch, custom_prompt, whisper_model):
    """Background job: transcribe → generate script → TTS → mix."""
    try:
        # Step 1: Transcribe (or use cached)
        jobs[cjob_id]['status'] = 'transcribing'
        jobs[cjob_id]['progress'] = 10
        jobs[cjob_id]['message'] = 'Transcribing video...'

        from core.transcriber import transcribe

        transcript = transcribe(video_path, model_size=whisper_model)

        if not transcript or not transcript.segments:
            jobs[cjob_id]['error'] = 'Transcription failed or empty'
            jobs[cjob_id]['status'] = 'error'
            return

        jobs[cjob_id]['progress'] = 30
        jobs[cjob_id]['message'] = f'Transcribed: {len(transcript.segments)} segments'

        # Step 2: Generate commentary script via LLM
        jobs[cjob_id]['status'] = 'generating'
        jobs[cjob_id]['progress'] = 40
        jobs[cjob_id]['message'] = 'AI generating narration script...'

        from ai.commentary import generate_commentary
        from core.llm_analyzer import LLMConfig

        llm_config = LLMConfig.from_env()
        transcript_dicts = [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in transcript.segments
        ]

        script = generate_commentary(
            transcript_segments=transcript_dicts,
            video_duration=transcript.duration,
            style=style,
            custom_prompt=custom_prompt,
            nvidia_api_key=llm_config.nvidia_api_key,
            nvidia_model=llm_config.nvidia_model,
        )

        jobs[cjob_id]['progress'] = 55
        jobs[cjob_id]['message'] = f'Script ready: {len(script.segments)} segments, ~{len(script.full_text.split())} words'
        jobs[cjob_id]['script'] = script.to_dict()

        # Step 3: Synthesize TTS
        jobs[cjob_id]['status'] = 'synthesizing'
        jobs[cjob_id]['progress'] = 65
        jobs[cjob_id]['message'] = 'Generating voiceover audio...'

        from media.tts_engine import synthesize_commentary

        # Output dir alongside the source video
        video_dir = os.path.dirname(video_path)
        video_stem = Path(video_path).stem
        commentary_dir = os.path.join(video_dir, f"{video_stem}_commentary")

        seg_dicts = [s.to_dict() for s in script.segments]
        tts_result = synthesize_commentary(
            commentary_segments=seg_dicts,
            output_dir=commentary_dir,
            voice_key=voice,
            rate=rate,
            pitch=pitch,
            video_duration=transcript.duration,
            narration_style=style,
        )

        if not tts_result.success:
            jobs[cjob_id]['error'] = f'TTS failed: {tts_result.error}'
            jobs[cjob_id]['status'] = 'error'
            return

        jobs[cjob_id]['progress'] = 80
        jobs[cjob_id]['message'] = f'Voice synthesized: {tts_result.duration:.1f}s audio'

        # Step 4: Mix audio with video
        jobs[cjob_id]['status'] = 'mixing'
        jobs[cjob_id]['progress'] = 85
        jobs[cjob_id]['message'] = f'Mixing audio ({mode})...'

        from media.audio_mixer import mix_commentary

        output_filename = f"{video_stem}_narrated.mp4"
        output_path = os.path.join(video_dir, output_filename)

        mix_result = mix_commentary(
            video_path=video_path,
            commentary_audio_path=tts_result.audio_path,
            output_path=output_path,
            mode=mode,
            original_volume=original_volume,
        )

        if not mix_result.success:
            jobs[cjob_id]['error'] = f'Audio mixing failed: {mix_result.error}'
            jobs[cjob_id]['status'] = 'error'
            return

        # Build result — determine serving URL
        # If this is a clip (in clips_output), build the URL accordingly
        result_data = {
            'output_path': output_path,
            'output_filename': output_filename,
            'duration': mix_result.duration,
            'mode': mode,
            'voice': voice,
            'script_segments': len(script.segments),
            'word_count': len(script.full_text.split()),
        }

        # Try to build a serveable URL
        clips_root = app.config.get('CLIPS_FOLDER', '')
        lib_root = app.config.get('LIBRARY_FOLDER', '')
        if clips_root and output_path.startswith(clips_root):
            rel = os.path.relpath(output_path, clips_root)
            parts = rel.replace('\\', '/').split('/')
            if len(parts) >= 2:
                result_data['url'] = f'/clips/{parts[-2]}/{parts[-1]}'
        elif lib_root and output_path.startswith(lib_root):
            result_data['url'] = f'/library-file/{output_filename}'

        jobs[cjob_id]['result'] = result_data
        jobs[cjob_id]['progress'] = 100
        jobs[cjob_id]['message'] = f'Commentary complete! {output_filename}'
        jobs[cjob_id]['status'] = 'completed'

    except Exception as e:
        logger.error(f"Commentary job failed: {e}", exc_info=True)
        jobs[cjob_id]['error'] = str(e)
        jobs[cjob_id]['status'] = 'error'


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

    # If no SRT exists, auto-transcribe the clip for context
    if not transcript_text:
        try:
            logger.info(f"No SRT found for {filename}, auto-transcribing for metadata context...")
            sub_result = generate_subtitles(clip_path, model_size='base', language='en')
            # Save the SRT for future use
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(sub_result['srt'])
            transcript_text = ' '.join(w['word'] for w in sub_result.get('words', []))
            logger.info(f"Auto-transcribed {len(sub_result.get('words', []))} words for metadata")
        except Exception as e:
            logger.warning(f"Auto-transcription for metadata failed: {e}")

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

    # ── Build content-aware metadata prompt ──
    # Detect language from transcript to generate in matching language
    has_hindi = any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in transcript_text[:500]) if transcript_text else False
    has_hinglish = bool(re.search(r'\b(kya|hai|kaise|kuch|mujhe|tumhe|pyaar|dil|nahi|lekin|yaar|bhai)\b',
                                   transcript_text[:500].lower())) if transcript_text else False
    detected_lang = 'Hindi/Hinglish' if (has_hindi or has_hinglish) else 'English'

    prompt = f"""You are a YouTube metadata expert. Generate accurate, content-faithful metadata for this video clip.

CRITICAL: Your titles and description MUST accurately reflect what happens in the video.
Do NOT invent drama, accusations, or events that are not in the transcript.
Do NOT use generic clickbait words (exposed, dark truth, banned, secret) unless the content actually contains them.

Clip info:
- Filename: {filename}
- Duration: {int(duration)} seconds
- Detected language: {detected_lang}
- Transcript: {transcript_text[:2000] if transcript_text else 'No transcript. Use filename for context.'}

RULES:
1. Read the transcript carefully. Understand what ACTUALLY happens in the clip.
2. Titles must describe the real content — if it's a romantic scene, say so. If it's a comedy bit, say so.
3. If the content is in Hindi/Hinglish, write titles in Hinglish (Roman script Hindi) with English mix.
   Example: "Main Aisa Kya Karu Ki Tumhe Pyaar Ho Jaye" | Emotional Love Scene
4. Titles under 70 characters. Make them compelling but HONEST.
5. Description: 2-3 sentences describing what happens + call to action + 3 relevant hashtags.
6. Tags: 12-15 tags that MATCH the actual content genre, language, and topic.
7. If it's a movie/show clip, include the movie name, actors, genre in tags.

Return ONLY valid JSON:
{{
    "titles": ["Title 1 (BEST)", "Title 2", "Title 3"],
    "title": "Your best title",
    "description": "Content-accurate description with hashtags",
    "tags": ["tag1", "tag2", "..."],
    "pinned_comment": "A genuine question about the content to drive discussion",
    "hook_script": "First 3 seconds hook that matches the actual content"
}}"""

    # ── Try Gemini first (best at content understanding) ──
    raw = None
    provider_used = None

    if llm_config.gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=llm_config.gemini_api_key)
            model = genai.GenerativeModel(llm_config.gemini_model)
            response = model.generate_content(prompt)
            raw = response.text.strip()
            provider_used = 'gemini'
        except Exception as e:
            logger.warning(f"Gemini metadata failed: {e}")

    # ── Fallback to NVIDIA ──
    if not raw and llm_config.nvidia_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=llm_config.nvidia_api_key,
            )
            response = client.chat.completions.create(
                model=llm_config.nvidia_model,
                messages=[
                    {"role": "system", "content": "Generate content-accurate YouTube metadata. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()
            provider_used = 'nvidia'
        except Exception as e:
            logger.warning(f"NVIDIA metadata failed: {e}")

    # ── Fallback to Groq ──
    if not raw and llm_config.groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=llm_config.groq_api_key)
            response = client.chat.completions.create(
                model=llm_config.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()
            provider_used = 'groq'
        except Exception as e:
            logger.warning(f"Groq metadata failed: {e}")

    if raw:
        try:
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3].strip()

            result = json.loads(raw)
            return jsonify({
                'title': result.get('title', filename)[:100],
                'titles': result.get('titles', [result.get('title', filename)])[:5],
                'description': result.get('description', '')[:5000],
                'tags': result.get('tags', [])[:15],
                'pinned_comment': result.get('pinned_comment', ''),
                'hook_script': result.get('hook_script', ''),
                'ai_generated': True,
                'provider': provider_used,
            })
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse {provider_used} metadata JSON: {e}")

    # ── All providers failed — basic fallback ──
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
        'warning': 'All AI providers failed, using basic metadata',
    })


@app.route('/api/youtube/generate-metadata-batch', methods=['POST'])
def youtube_generate_metadata_batch():
    """Generate AI metadata for all clips in a job at once."""
    data = request.get_json() or {}
    job_id = data.get('job_id', '')

    if not job_id:
        return jsonify({'error': 'Missing job_id'}), 400

    clip_dir = _find_job_dir(job_id)
    if not os.path.isdir(clip_dir):
        return jsonify({'error': 'Clip directory not found'}), 404

    clip_files = sorted([
        f for f in os.listdir(clip_dir)
        if f.lower().endswith(('.mp4', '.webm', '.mov'))
        and '_subtitled' not in f.lower()
        and '_overlay' not in f.lower()
        and '_thumb_' not in f.lower()
        and '_mod.' not in f.lower()
        and '_narrated.' not in f.lower()
    ])

    if not clip_files:
        return jsonify({'error': 'No clips found'}), 404

    results = []
    for idx, filename in enumerate(clip_files):
        clip_path = os.path.join(clip_dir, filename)

        # Read SRT transcript if available
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

        duration = _get_video_duration(clip_path)

        # Generate content-aware metadata via LLM (Gemini → NVIDIA → Groq)
        llm_config = LLMConfig.from_env()
        ai_generated = False
        title, description, tags = filename, '', []

        # Detect language
        has_hindi = any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in transcript_text[:300]) if transcript_text else False
        has_hinglish = bool(re.search(r'\b(kya|hai|kaise|mujhe|tumhe|pyaar|dil|nahi)\b',
                                       transcript_text[:300].lower())) if transcript_text else False
        det_lang = 'Hindi/Hinglish' if (has_hindi or has_hinglish) else 'English'

        batch_prompt = f"""Generate accurate, content-faithful YouTube metadata for clip {idx+1}/{len(clip_files)}.
Read the transcript and describe what ACTUALLY happens. Do NOT invent drama or use generic clickbait.
If content is in Hindi/Hinglish, write titles in Hinglish (Roman script).

Clip: {filename} | Duration: {int(duration)}s | Language: {det_lang}
Transcript: {transcript_text[:1200] if transcript_text else 'No transcript. Use filename.'}

Return ONLY valid JSON:
{{"titles": ["Title 1", "Title 2", "Title 3"], "title": "Best title",
"description": "Accurate description + 3 hashtags", "tags": ["12-15 content-matching tags"],
"pinned_comment": "Genuine question about the content", "hook_script": "Content-accurate hook"}}"""

        # Try Gemini first
        raw_batch = None
        if not raw_batch and llm_config.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=llm_config.gemini_api_key)
                model = genai.GenerativeModel(llm_config.gemini_model)
                resp = model.generate_content(batch_prompt)
                raw_batch = resp.text.strip()
            except Exception as e:
                logger.warning(f"Gemini batch metadata failed for {filename}: {e}")

        # Fallback to NVIDIA
        if not raw_batch and llm_config.nvidia_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=llm_config.nvidia_api_key,
                )
                response = client.chat.completions.create(
                    model=llm_config.nvidia_model,
                    messages=[{"role": "user", "content": batch_prompt}],
                    temperature=0.5,
                    max_tokens=1024,
                )
                raw_batch = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"NVIDIA batch metadata failed for {filename}: {e}")

        if raw_batch:
            try:
                raw = raw_batch
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3].strip()

                parsed = json.loads(raw)
                title = parsed.get('title', filename)[:100]
                titles = parsed.get('titles', [title])[:5]
                description = parsed.get('description', '')[:5000]
                tags = parsed.get('tags', [])[:15]
                pinned_comment = parsed.get('pinned_comment', '')
                hook_script = parsed.get('hook_script', '')
                ai_generated = True
            except Exception as e:
                logger.warning(f"AI metadata failed for {filename}: {e}")

        if not ai_generated:
            meta = youtube_uploader.generate_clip_metadata(
                clip_info={'hook_text': transcript_text[:200], 'duration': duration},
                source_title=filename,
                clip_number=idx + 1, total_clips=len(clip_files),
            )
            title, description, tags = meta['title'], meta['description'], meta['tags']
            titles, pinned_comment, hook_script = [title], '', ''

        results.append({
            'filename': filename,
            'title': title,
            'titles': titles if ai_generated else [title],
            'description': description,
            'tags': tags,
            'pinned_comment': pinned_comment if ai_generated else '',
            'hook_script': hook_script if ai_generated else '',
            'ai_generated': ai_generated,
        })

    return jsonify({'clips': results, 'total': len(results)})


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

    # Auto-append credit line if source video has uploader info
    credit = _get_credit_line(job_id=job_id)
    if credit and credit.strip() not in description:
        description += credit

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
        and '_mod.' not in f.lower()
        and '_narrated.' not in f.lower()
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
              source_title, custom_tags, made_for_kids, clips_meta, job_id),
        daemon=True,
    )
    thread.start()
    return jsonify({'upload_job_id': upload_job_id, 'total': len(clip_files)})


def _youtube_upload_job(upload_job_id, clip_dir, clip_files, privacy, category,
                        source_title, custom_tags, made_for_kids, clips_meta,
                        job_id=""):
    """Background: upload all clips to YouTube one by one."""
    total = len(clip_files)
    results = []

    for i, filename in enumerate(clip_files):
        file_path = os.path.join(clip_dir, filename)
        jobs[upload_job_id]['current_file'] = filename
        jobs[upload_job_id]['message'] = f'Uploading {i+1}/{total}: {filename}'

        # Get metadata — either from AI-generated clips_meta or auto-generate
        clip_info = {}
        if clips_meta and i < len(clips_meta):
            clip_info = clips_meta[i]

        # If AI-generated metadata is present, use it directly
        if clip_info.get('title'):
            title = clip_info['title']
            description = clip_info.get('description', '')
            clip_tags = clip_info.get('tags', [])
            if isinstance(clip_tags, str):
                clip_tags = [t.strip() for t in clip_tags.split(',') if t.strip()]
        else:
            meta = youtube_uploader.generate_clip_metadata(
                clip_info=clip_info,
                source_title=source_title,
                clip_number=i + 1,
                total_clips=total,
            )
            title = meta['title']
            description = meta['description']
            clip_tags = meta['tags']

        # Merge in extra tags from the form
        if custom_tags:
            clip_tags = list(set(clip_tags + custom_tags))[:15]

        # Auto-append credit line from source video
        credit = _get_credit_line(job_id=job_id)
        if credit and credit.strip() not in description:
            description += credit

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
            tags=clip_tags,
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


# NOTE: Legacy /api/process route removed — it created orphan directories
# and always returned 400. Use the Video Library panel workflow instead.


# ═══════════════════════════════════════════════════
#  ENGAGEMENT CLIPS (5-20 min segments from long videos)
# ═══════════════════════════════════════════════════

@app.route('/api/engagement-clip', methods=['POST'])
def engagement_clip():
    """Extract medium-length (5-20 min) high-engagement segments from a video."""
    video_id = request.form.get('video_id', '')
    entry = video_library.get_video(video_id) if video_id else None
    if not entry:
        return jsonify({'error': 'Select a video from the library'}), 400
    if not Path(entry.file_path).exists():
        return jsonify({'error': 'Video file missing'}), 400

    settings = {
        'model_size': request.form.get('model_size', 'base'),
        'max_segments': int(request.form.get('max_segments', 5)),
        'min_duration': int(request.form.get('min_duration', 300)),
        'max_duration': int(request.form.get('max_duration', 1200)),
        'target_duration': int(request.form.get('target_duration', 600)),
        'min_score': float(request.form.get('min_score', 0.35)),
        'crop_vertical': request.form.get('crop_vertical', 'true') == 'true',
        'use_llm': request.form.get('use_llm', 'true') == 'true',
    }

    job_id = str(uuid.uuid4())[:8]
    job_dir = _make_clip_output_dir(entry.title, job_id)

    jobs[job_id] = {
        'status': 'transcribing',
        'progress': 5,
        'message': 'Starting engagement analysis...',
        'clips': [],
        'error': None,
    }

    thread = threading.Thread(
        target=_engagement_clip_job,
        args=(job_id, entry.file_path, video_id, job_dir, settings),
        daemon=True,
    )
    thread.start()
    return jsonify({'job_id': job_id, 'status': 'started'})


def _engagement_clip_job(job_id, video_path, video_id, output_dir, settings):
    """Background: transcribe → engagement analyze → extract segments."""
    try:
        # Transcribe
        jobs[job_id]['status'] = 'transcribing'
        jobs[job_id]['message'] = f'Transcribing video ({settings["model_size"]})...'
        jobs[job_id]['progress'] = 10

        transcript = transcribe(
            video_path=video_path,
            model_size=settings['model_size'],
        )
        jobs[job_id]['message'] = f'Transcribed: {len(transcript.segments)} segments'
        jobs[job_id]['progress'] = 40
        transcript.save(os.path.join(output_dir, 'transcript.json'))

        # Engagement analysis
        jobs[job_id]['status'] = 'analyzing'
        jobs[job_id]['message'] = 'Analyzing for high-engagement segments...'
        jobs[job_id]['progress'] = 45

        eng_config = EngagementConfig(
            min_segment_duration=settings['min_duration'],
            max_segment_duration=settings['max_duration'],
            target_segment_duration=settings['target_duration'],
            max_segments=settings['max_segments'],
            min_engagement_score=settings['min_score'],
            crop_vertical=settings['crop_vertical'],
            use_llm=settings['use_llm'],
        )

        # Try LLM-assisted analysis first
        llm_segments = []
        if settings.get('use_llm'):
            jobs[job_id]['message'] = 'Running AI engagement analysis...'
            jobs[job_id]['progress'] = 50
            llm_segments = analyze_with_llm_for_engagement(transcript, eng_config)

        analyzer = EngagementAnalyzer(eng_config)
        candidates = analyzer.analyze(transcript, llm_segments=llm_segments)

        if not candidates:
            jobs[job_id]['message'] = 'No high-engagement segments found. Try lowering min score.'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['status'] = 'completed'
            return

        jobs[job_id]['message'] = f'{len(candidates)} segments found'
        jobs[job_id]['progress'] = 60

        # Extract segments using FFmpeg
        jobs[job_id]['status'] = 'splitting'
        total = len(candidates)

        clipper_config = ClipperConfig(
            min_clip_duration=settings['min_duration'],
            max_clip_duration=settings['max_duration'],
            crop_vertical=settings['crop_vertical'],
            anti_copyright=settings.get('anti_copyright', True),
            pre_hook_padding=0,  # engagement segments are already bounded
            post_hook_padding=0,
        )
        clipper = VideoClipper(clipper_config)

        clips_info = []
        for i, seg in enumerate(candidates):
            jobs[job_id]['message'] = f'Extracting segment {i+1} of {total}...'
            jobs[job_id]['progress'] = 60 + int((i / total) * 35)

            # Create a ClipCandidate for the clipper
            from core.analyzer import ClipCandidate
            cand = ClipCandidate(
                start=seg.start, end=seg.end,
                score=seg.score, hook_text=seg.summary,
                reason=seg.reason,
            )
            results = clipper.extract_all_clips(
                video_path=video_path,
                candidates=[cand],
                output_dir=output_dir,
                video_duration=transcript.duration,
            )
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
                        'hook_text': seg.title_hint,
                        'summary': seg.summary,
                    })

        video_library.add_clips_directory(video_id, output_dir)

        jobs[job_id]['clips'] = clips_info
        jobs[job_id]['message'] = f'Done! {len(clips_info)} engagement clips extracted.'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['status'] = 'completed'

    except Exception as e:
        logger.exception(f"Engagement clip job {job_id} failed")
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['message'] = f'Error: {e}'


# ═══════════════════════════════════════════════════
#  FULL VIDEO PROCESSING (any length, all features)
# ═══════════════════════════════════════════════════

@app.route('/api/full-video', methods=['POST'])
def full_video_process():
    """Process a full-length video with all shorts/reels features."""
    video_id = request.form.get('video_id', '')
    entry = video_library.get_video(video_id) if video_id else None
    if not entry:
        return jsonify({'error': 'Select a video from the library'}), 400
    if not Path(entry.file_path).exists():
        return jsonify({'error': 'Video file missing'}), 400

    config = FullVideoConfig(
        crop_vertical=request.form.get('crop_vertical', 'true') == 'true',
        enable_subtitles=request.form.get('enable_subtitles', 'false') == 'true',
        subtitle_model_size=request.form.get('subtitle_model_size', 'base'),
        subtitle_language=request.form.get('subtitle_language') or None,
        enable_overlay=request.form.get('enable_overlay', 'false') == 'true',
        overlay_text=request.form.get('overlay_text', ''),
        enable_thumbnail=request.form.get('enable_thumbnail', 'false') == 'true',
        thumbnail_title=request.form.get('thumbnail_title', ''),
        thumbnail_style=request.form.get('thumbnail_style', 'bold'),
        enable_modulation=request.form.get('enable_modulation', 'false') == 'true',
        modulation_preset=request.form.get('modulation_preset', ''),
    )

    job_id = str(uuid.uuid4())[:8]
    job_dir = _make_clip_output_dir(entry.title, job_id)

    jobs[job_id] = {
        'status': 'processing',
        'progress': 5,
        'message': 'Starting full video processing...',
        'result': None,
        'error': None,
    }

    thread = threading.Thread(
        target=_full_video_job,
        args=(job_id, entry.file_path, video_id, job_dir, config),
        daemon=True,
    )
    thread.start()
    return jsonify({'job_id': job_id, 'status': 'started'})


def _full_video_job(job_id, video_path, video_id, output_dir, config):
    """Background: full video processing pipeline."""
    try:
        def _progress(pct, msg):
            jobs[job_id]['progress'] = pct
            jobs[job_id]['message'] = msg

        result = process_full_video(
            video_path=video_path,
            output_dir=output_dir,
            config=config,
            progress_callback=_progress,
        )

        video_library.add_clips_directory(video_id, output_dir)

        if result.success:
            fname = Path(result.output_path).name
            result_data = {
                'output_url': f'/clips/{job_id}/{fname}',
                'output_filename': fname,
                'processing_steps': result.processing_steps,
                'duration': result.duration,
            }
            if result.thumbnail_path:
                tname = Path(result.thumbnail_path).name
                result_data['thumbnail_url'] = f'/clips/{job_id}/{tname}'
            if result.srt_path:
                sname = Path(result.srt_path).name
                result_data['srt_url'] = f'/clips/{job_id}/{sname}'

            jobs[job_id]['result'] = result_data
            jobs[job_id]['message'] = f'Done! {len(result.processing_steps)} steps applied.'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['status'] = 'completed'
        else:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = result.error
            jobs[job_id]['message'] = f'Error: {result.error}'

    except Exception as e:
        logger.exception(f"Full video job {job_id} failed")
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['message'] = f'Error: {e}'


# ═══════════════════════════════════════════════════
#  CONTENT FACTORY — AI Video Generation from Trends
# ═══════════════════════════════════════════════════

@app.route('/factory')
def factory_page():
    """Serve the Content Factory UI — completely separate from clipper."""
    return render_template('factory.html')


@app.route('/api/factory/trending', methods=['GET'])
def api_factory_trending():
    """Fetch trending topics from YouTube + Google Trends + Reddit."""
    try:
        region = request.args.get('region', 'US')
        category = request.args.get('category', '24')
        max_results = int(request.args.get('max', 30))

        topics = scout_trending(
            region=region,
            max_per_source=max_results // 3 + 1,
        )

        return jsonify({
            'success': True,
            'topics': topics,
            'count': len(topics),
            'categories': get_trend_categories(),
        })
    except Exception as e:
        logger.error(f"Trending fetch error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/factory/generate', methods=['POST'])
def api_factory_generate():
    """Start a content generation job from a trending topic."""
    try:
        data = request.get_json() or {}
        topic = data.get('topic', '').strip()
        if not topic:
            return jsonify({'success': False, 'error': 'Topic is required'}), 400

        output_dir = os.path.join(app.config['CLIPS_FOLDER'], 'factory')

        job_id = factory_start(
            topic=topic,
            topic_description=data.get('description', ''),
            topic_keywords=data.get('keywords', []),
            target_duration=int(data.get('target_duration', 45)),
            tone=data.get('tone', 'energetic'),
            style=data.get('style', 'informative'),
            music_mood=data.get('music_mood', 'upbeat'),
            music_volume=float(data.get('music_volume', 0.15)),
            color_grade=data.get('color_grade', 'cinematic'),
            output_base_dir=output_dir,
        )

        return jsonify({'success': True, 'job_id': job_id})

    except Exception as e:
        logger.error(f"Factory generate error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/factory/jobs', methods=['GET'])
def api_factory_jobs():
    """List all content factory jobs."""
    try:
        jobs = factory_list_jobs()
        return jsonify({'success': True, 'jobs': jobs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/factory/job/<job_id>', methods=['GET'])
def api_factory_job_status(job_id):
    """Get status of a specific factory job."""
    try:
        job = factory_get_job(job_id)
        if not job:
            return jsonify({'success': False, 'error': 'Job not found'}), 404
        return jsonify({'success': True, 'job': job.to_dict()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/factory/job/<job_id>/video')
def api_factory_serve_video(job_id):
    """Serve the final generated video."""
    try:
        job = factory_get_job(job_id)
        if not job or not job.final_path or not os.path.isfile(job.final_path):
            return jsonify({'success': False, 'error': 'Video not found'}), 404

        directory = os.path.dirname(job.final_path)
        filename = os.path.basename(job.final_path)
        return send_from_directory(directory, filename, mimetype='video/mp4')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/factory/job/<job_id>', methods=['DELETE'])
def api_factory_delete_job(job_id):
    """Delete a factory job and its output."""
    try:
        success = factory_cleanup_job(job_id)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/factory/config', methods=['GET'])
def api_factory_config():
    """Return configuration options for the factory UI."""
    return jsonify({
        'success': True,
        'styles': get_script_styles(),
        'music_moods': get_mood_options(),
        'categories': get_trend_categories(),
    })


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
            serve(app, host="0.0.0.0", port=port, threads=8,
                  channel_timeout=600,
                  recv_bytes=262144,
                  max_request_body_size=5368709120,
                  max_request_header_size=65536,
                  )
        except ImportError:
            logger.warning("Waitress not installed, falling back to Flask dev server")
            app.run(host="0.0.0.0", port=port, debug=False)
