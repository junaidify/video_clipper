"""
Web Context & Shared State Module
Holds shared state, directories, singletons (VideoLibrary, PatternTrainer),
job tracking registries, and concurrency locks.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

from video_clipper.media_management.library import VideoLibrary
from video_clipper.pattern_learning.trainer import PatternTrainer

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_FOLDER = str(BASE_DIR / "uploads")
OUTPUT_FOLDER = str(BASE_DIR / "clips_output")
LIBRARY_FOLDER = str(BASE_DIR / "video_library")
TRAINING_FOLDER = str(BASE_DIR / "training_sessions")
FACTORY_FOLDER = str(BASE_DIR / "factory_output")

# Ensure required runtime directories exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, LIBRARY_FOLDER, TRAINING_FOLDER, FACTORY_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Shared singletons
_video_library: Optional[VideoLibrary] = None
_pattern_trainer: Optional[PatternTrainer] = None
_init_lock = threading.Lock()

# Thread pools and job registries
executor = ThreadPoolExecutor(max_workers=4)

jobs_lock = threading.Lock()
jobs: Dict[str, dict] = {}

_upload_status_lock = threading.Lock()
_upload_status: Dict[str, dict] = {}

_full_video_jobs_lock = threading.Lock()
_full_video_jobs: Dict[str, dict] = {}


def get_video_library() -> VideoLibrary:
    global _video_library
    with _init_lock:
        if _video_library is None:
            _video_library = VideoLibrary(LIBRARY_FOLDER)
        return _video_library


def get_pattern_trainer() -> PatternTrainer:
    global _pattern_trainer
    with _init_lock:
        if _pattern_trainer is None:
            _pattern_trainer = PatternTrainer(TRAINING_FOLDER)
        return _pattern_trainer


def set_job(job_id: str, **kwargs):
    with jobs_lock:
        if job_id not in jobs:
            jobs[job_id] = {}
        jobs[job_id].update(kwargs)


def get_job_state(job_id: str) -> Optional[dict]:
    with jobs_lock:
        job = jobs.get(job_id)
        return dict(job) if job else None


def set_upload_job(job_id: str, **kwargs):
    with _upload_status_lock:
        if job_id not in _upload_status:
            _upload_status[job_id] = {}
        _upload_status[job_id].update(kwargs)


def get_upload_job_state(job_id: str) -> Optional[dict]:
    with _upload_status_lock:
        job = _upload_status.get(job_id)
        return dict(job) if job else None


def set_full_video_job(job_id: str, **kwargs):
    with _full_video_jobs_lock:
        if job_id not in _full_video_jobs:
            _full_video_jobs[job_id] = {}
        _full_video_jobs[job_id].update(kwargs)


def get_full_video_job_state(job_id: str) -> Optional[dict]:
    with _full_video_jobs_lock:
        job = _full_video_jobs.get(job_id)
        return dict(job) if job else None
