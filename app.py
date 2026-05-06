"""
Flask Web Application for Video Auto-Clipper
Upload a video or paste a YouTube URL → analyze → split into short clips.
"""
import json
import logging
import os
import sys
import time
import uuid
import threading
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
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
from downloader import is_valid_url, download_video, get_video_info
from llm_analyzer import analyze_with_llm, LLMConfig

# ─── App Setup ───
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['CLIPS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'clips_output')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'video-clipper-dev-key')

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'm4v'}

# In-memory job tracker
jobs = {}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Routes ───

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/check-url', methods=['POST'])
def check_url():
    """Check if a URL is valid and get video info."""
    data = request.get_json()
    url = data.get('url', '').strip()

    if not url or not is_valid_url(url):
        return jsonify({'valid': False, 'error': 'Invalid or unsupported URL'})

    try:
        info = get_video_info(url)
        return jsonify({'valid': True, 'info': info})
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)})


@app.route('/api/process', methods=['POST'])
def process_video():
    """
    Start video processing job.
    Accepts either file upload or URL.
    Returns job_id for progress tracking.
    """
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(app.config['CLIPS_FOLDER'], job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Parse settings from form
    settings = {
        'model_size': request.form.get('model_size', 'base'),
        'max_clips': int(request.form.get('max_clips', 10)),
        'min_score': float(request.form.get('min_score', 0.4)),
        'min_duration': int(request.form.get('min_duration', 15)),
        'max_duration': int(request.form.get('max_duration', 60)),
        'crop_vertical': request.form.get('crop_vertical', 'true') == 'true',
        'use_llm': request.form.get('use_llm', 'false') == 'true',
    }

    video_path = None
    source_type = None

    # Check for URL input
    url = request.form.get('url', '').strip()
    if url and is_valid_url(url):
        source_type = 'url'
        jobs[job_id] = {
            'status': 'downloading',
            'progress': 0,
            'message': 'Downloading video...',
            'url': url,
            'clips': [],
            'error': None,
        }
        # Start processing in background thread
        thread = threading.Thread(
            target=_process_job,
            args=(job_id, None, url, source_type, job_dir, settings),
            daemon=True,
        )
        thread.start()
        return jsonify({'job_id': job_id, 'status': 'started'})

    # Check for file upload
    if 'video' in request.files:
        file = request.files['video']
        if file and file.filename and allowed_file(file.filename):
            source_type = 'upload'
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
            file.save(video_path)

            jobs[job_id] = {
                'status': 'processing',
                'progress': 0,
                'message': 'Video uploaded, starting analysis...',
                'clips': [],
                'error': None,
            }

            thread = threading.Thread(
                target=_process_job,
                args=(job_id, video_path, None, source_type, job_dir, settings),
                daemon=True,
            )
            thread.start()
            return jsonify({'job_id': job_id, 'status': 'started'})

    return jsonify({'error': 'No valid video file or URL provided'}), 400


@app.route('/api/status/<job_id>')
def job_status(job_id):
    """Get processing status for a job."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(jobs[job_id])


@app.route('/clips/<job_id>/<filename>')
def serve_clip(job_id, filename):
    """Serve a generated clip file."""
    clip_dir = os.path.join(app.config['CLIPS_FOLDER'], job_id)
    return send_from_directory(clip_dir, filename)


@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Return current API key status and system dependency checks."""
    import shutil
    config = LLMConfig.from_env()
    return jsonify({
        'groq_configured': bool(config.groq_api_key),
        'gemini_configured': bool(config.gemini_api_key),
        'nvidia_configured': bool(config.nvidia_api_key),
        'llm_available': config.has_any_key(),
        'ffmpeg_installed': shutil.which('ffmpeg') is not None,
    })


# ─── Background Processing ───

def _process_job(job_id: str, video_path: str, url: str,
                 source_type: str, output_dir: str, settings: dict):
    """Run the full pipeline in a background thread."""
    try:
        # Step 1: Download if URL
        if source_type == 'url':
            jobs[job_id]['status'] = 'downloading'
            jobs[job_id]['message'] = 'Downloading video from URL...'
            jobs[job_id]['progress'] = 5

            dl_result = download_video(url, app.config['UPLOAD_FOLDER'])
            if not dl_result.success:
                jobs[job_id]['status'] = 'error'
                jobs[job_id]['error'] = f'Download failed: {dl_result.error}'
                return

            video_path = dl_result.file_path
            jobs[job_id]['message'] = f'Downloaded: {dl_result.title}'
            jobs[job_id]['progress'] = 15

        # Step 2: Transcribe
        jobs[job_id]['status'] = 'transcribing'
        jobs[job_id]['message'] = f'Transcribing audio (model: {settings["model_size"]})...'
        jobs[job_id]['progress'] = 20

        transcript = transcribe(
            video_path=video_path,
            model_size=settings['model_size'],
        )

        jobs[job_id]['message'] = f'Transcribed: {len(transcript.segments)} segments'
        jobs[job_id]['progress'] = 50

        # Save transcript
        transcript.save(os.path.join(output_dir, 'transcript.json'))

        # Step 3: Analyze
        jobs[job_id]['status'] = 'analyzing'
        jobs[job_id]['message'] = 'Analyzing content for hooks...'
        jobs[job_id]['progress'] = 55

        analyzer_config = AnalyzerConfig(
            min_hook_score=settings['min_score'],
            max_clips=settings['max_clips'],
        )
        analyzer = ContentAnalyzer(analyzer_config)
        candidates = analyzer.analyze(transcript)

        # Optional: LLM fallback
        if settings.get('use_llm') and len(candidates) < 2:
            jobs[job_id]['message'] = 'NLP found few hooks, trying LLM analysis...'
            llm_candidates = analyze_with_llm(transcript)
            if llm_candidates:
                # Convert LLM results to ClipCandidate objects
                for lc in llm_candidates:
                    candidates.append(ClipCandidate(
                        start=lc['start'],
                        end=lc['end'],
                        score=lc['score'],
                        hook_text=lc['hook_text'],
                        reason=f"llm_{lc['reason']}",
                    ))
                # Re-sort and deduplicate
                candidates.sort(key=lambda c: c.score, reverse=True)
                candidates = candidates[:settings['max_clips']]

        if not candidates:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['message'] = 'No hook-worthy moments found. Try lowering the minimum score.'
            jobs[job_id]['progress'] = 100
            return

        jobs[job_id]['message'] = f'Found {len(candidates)} clip candidates'
        jobs[job_id]['progress'] = 65

        # Step 4: Split video
        jobs[job_id]['status'] = 'splitting'
        jobs[job_id]['message'] = 'Splitting video into clips...'

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

        # Build clip info for frontend
        clips_info = []
        for r in results:
            if r.success:
                filename = Path(r.output_path).name
                clips_info.append({
                    'clip_number': r.clip_number,
                    'filename': filename,
                    'url': f'/clips/{job_id}/{filename}',
                    'start': r.start,
                    'end': r.end,
                    'duration': r.duration,
                    'score': r.score,
                    'reason': r.reason,
                    'hook_text': r.hook_text[:200],
                })

        # Done
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['clips'] = clips_info
        jobs[job_id]['message'] = f'Done! {len(clips_info)} clips created.'
        jobs[job_id]['progress'] = 100

        logger.info(f"Job {job_id} completed: {len(clips_info)} clips")

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['message'] = f'Error: {str(e)}'


# ─── Main ───

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CLIPS_FOLDER'], exist_ok=True)

    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'

    print(f"""
╔══════════════════════════════════════════════════╗
║           VIDEO AUTO-CLIPPER  (Web UI)           ║
║   Upload video or paste YouTube URL              ║
║   http://localhost:{port}                          ║
╚══════════════════════════════════════════════════╝
    """)

    app.run(host='0.0.0.0', port=port, debug=debug)
