"""
Web Job Manager Module
Coordinates background asynchronous tasks for smart clipping, full-video processing,
and batch YouTube uploads.
"""
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional, List, Dict

from video_clipper.config import PipelineConfig, TranscriberConfig, AnalyzerConfig, ClipperConfig
from video_clipper.clipping.transcriber import transcribe
from video_clipper.clipping.analyzer import ContentAnalyzer
from video_clipper.clipping.clipper import VideoClipper
from video_clipper.clipping.full_video_processor import process_full_video, FullVideoConfig
from video_clipper.distribution.youtube_uploader import (
    upload_video, upload_caption, generate_clip_metadata
)
from video_clipper.media_management.downloader import download_video
from video_clipper.web.context import (
    executor,
    set_job,
    get_job_state,
    set_upload_job,
    set_full_video_job,
    get_video_library,
    get_pattern_trainer,
    OUTPUT_FOLDER,
    UPLOAD_FOLDER,
)

logger = logging.getLogger(__name__)


def start_smart_clip_job(
    job_id: str,
    video_source_type: str,     # 'upload', 'url', 'library'
    video_source_value: str,    # file path, url, or video_id
    pipeline_config: PipelineConfig,
    anti_copyright: bool = False,
    anti_copyright_config: Optional[dict] = None,
    session_id: Optional[str] = None,
    title: Optional[str] = None,
):
    """Launch async smart clip extraction job."""
    set_job(
        job_id,
        id=job_id,
        status="running",
        progress=0.0,
        stage="init",
        message="Initializing job...",
        error=None,
        clips=[],
        transcript=None,
        analysis=None,
        start_time=time.time(),
    )

    executor.submit(
        _run_smart_clip_job,
        job_id,
        video_source_type,
        video_source_value,
        pipeline_config,
        anti_copyright,
        anti_copyright_config,
        session_id,
        title,
    )


def _run_smart_clip_job(
    job_id: str,
    video_source_type: str,
    video_source_value: str,
    config: PipelineConfig,
    anti_copyright: bool,
    anti_copyright_config: Optional[dict],
    session_id: Optional[str],
    custom_title: Optional[str],
):
    try:
        library = get_video_library()
        video_path = None
        video_title = custom_title or "video"
        library_entry = None

        # 1. Resolve source
        if video_source_type == "library":
            entry = library.get_video(video_source_value)
            if not entry or not os.path.isfile(entry.file_path):
                raise FileNotFoundError(f"Library video '{video_source_value}' not found.")
            video_path = entry.file_path
            video_title = entry.title
            library_entry = entry
        elif video_source_type == "url":
            set_job(job_id, stage="download", message="Downloading video from URL...", progress=5.0)

            def dl_progress(pct, msg):
                set_job(job_id, progress=round(pct * 0.25, 1), message=msg)

            dl_result = download_video(
                video_source_value,
                output_dir=UPLOAD_FOLDER,
                progress_callback=dl_progress,
            )
            if not dl_result.success or not dl_result.file_path:
                raise RuntimeError(f"Download failed: {dl_result.error}")

            video_path = dl_result.file_path
            video_title = dl_result.title or custom_title or "Downloaded Video"

            # Add to library
            library_entry = library.add_video(
                source_path=video_path,
                title=video_title,
                source="url",
                source_url=video_source_value,
                duration=dl_result.duration,
                uploader=dl_result.uploader,
                channel_url=dl_result.channel_url,
            )
        else:
            video_path = video_source_value
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Uploaded video '{video_path}' not found.")

        # Prepare job output directory
        safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_title)[:40]
        job_output_dir = os.path.join(OUTPUT_FOLDER, f"{safe_title}_{job_id}")
        os.makedirs(job_output_dir, exist_ok=True)

        if library_entry:
            library.add_clips_directory(library_entry.video_id, job_output_dir)

        # 2. Transcription
        set_job(job_id, stage="transcribe", message="Extracting audio & transcribing speech...", progress=30.0)
        cache_dir = os.path.join(job_output_dir, "cache")
        transcript = transcribe(video_path, config=config.transcriber, cache_dir=cache_dir)
        set_job(job_id, transcript=transcript.to_dict(), progress=60.0)

        # 3. Content Analysis
        set_job(job_id, stage="analyze", message="Analyzing transcript for viral hooks...", progress=65.0)
        creator_profile = None
        if session_id:
            trainer = get_pattern_trainer()
            profile = trainer.get_profile(session_id)
            if profile:
                creator_profile = profile.to_dict()

        analyzer = ContentAnalyzer(config=config.analyzer, creator_profile=creator_profile)
        candidates = analyzer.analyze(transcript)

        if not candidates:
            set_job(
                job_id,
                status="completed",
                progress=100.0,
                stage="done",
                message="No moments met the viral score threshold.",
                clips=[],
                analysis=[],
            )
            return

        set_job(job_id, analysis=[c.to_dict() for c in candidates], progress=75.0)

        # 4. Cutting & Video Export
        set_job(job_id, stage="clip", message=f"Rendering {len(candidates)} clips via FFmpeg...", progress=78.0)

        clipper = VideoClipper(config=config.clipper)
        total_cands = len(candidates)

        def clip_progress(curr, total, msg):
            pct = 78.0 + (curr / total) * 20.0
            set_job(job_id, progress=round(pct, 1), message=msg)

        clip_results = clipper.process_clips(
            video_path,
            candidates=candidates,
            output_dir=job_output_dir,
            anti_copyright=anti_copyright,
            anti_copyright_config=anti_copyright_config,
            progress_callback=clip_progress,
        )

        clips_data = [r.to_dict() for r in clip_results]

        # Save analysis JSON
        if config.save_analysis:
            analysis_file = os.path.join(job_output_dir, "analysis.json")
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump({
                    "job_id": job_id,
                    "video_title": video_title,
                    "transcript": transcript.to_dict(),
                    "candidates": [c.to_dict() for c in candidates],
                    "clips": clips_data,
                }, f, indent=2)

        set_job(
            job_id,
            status="completed",
            progress=100.0,
            stage="done",
            message=f"Successfully generated {len(clips_data)} clips!",
            clips=clips_data,
        )
        logger.info(f"Smart clip job [{job_id}] completed successfully.")

    except Exception as e:
        logger.exception(f"Smart clip job [{job_id}] failed: {e}")
        set_job(
            job_id,
            status="failed",
            error=str(e),
            message=f"Processing failed: {e}",
            stage="error",
        )


def start_full_video_job(
    job_id: str,
    video_path: str,
    output_dir: str,
    config: FullVideoConfig,
):
    """Launch async full video enhancement job."""
    set_full_video_job(
        job_id,
        id=job_id,
        status="running",
        progress=0.0,
        message="Starting full video processing...",
        error=None,
        result=None,
    )

    def _worker():
        try:
            def _prog(pct, msg):
                set_full_video_job(job_id, progress=float(pct), message=msg)

            res = process_full_video(video_path, output_dir, config, progress_callback=_prog)
            if res.success:
                set_full_video_job(job_id, status="completed", progress=100.0, message="Processing complete!", result=res.to_dict())
            else:
                set_full_video_job(job_id, status="failed", error=res.error, message=f"Failed: {res.error}")
        except Exception as e:
            set_full_video_job(job_id, status="failed", error=str(e), message=f"Failed: {e}")

    executor.submit(_worker)


def start_youtube_upload_batch(
    job_id: str,
    clips: List[dict],
    source_title: str = "",
    category: str = "entertainment",
    privacy: str = "private",
    made_for_kids: bool = False,
    schedule_interval_hours: float = 0.0,
):
    """Launch async YouTube batch upload job."""
    set_upload_job(
        job_id,
        job_id=job_id,
        status="uploading",
        total=len(clips),
        uploaded=0,
        progress_pct=0,
        results=[],
        error=None,
    )

    executor.submit(
        _run_youtube_upload_batch,
        job_id,
        clips,
        source_title,
        category,
        privacy,
        made_for_kids,
        schedule_interval_hours,
    )


def _run_youtube_upload_batch(
    job_id: str,
    clips: List[dict],
    source_title: str,
    category: str,
    privacy: str,
    made_for_kids: bool,
    schedule_interval_hours: float,
):
    from datetime import datetime, timedelta, timezone

    results = []
    total = len(clips)
    base_time = datetime.now(timezone.utc) if schedule_interval_hours > 0 else None

    for i, clip in enumerate(clips, 1):
        file_path = clip.get("file_path") or clip.get("output_path", "")
        filename = Path(file_path).name

        set_upload_job(
            job_id,
            current_file=filename,
            progress_pct=int((i - 1) / total * 100),
            uploaded=i - 1,
        )

        metadata = generate_clip_metadata(clip, source_title=source_title, clip_number=i, total_clips=total)

        publish_at = None
        if base_time and schedule_interval_hours > 0:
            sched_time = base_time + timedelta(hours=schedule_interval_hours * (i - 1))
            publish_at = sched_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        res = upload_video(
            file_path=file_path,
            title=metadata["title"],
            description=metadata["description"],
            tags=metadata["tags"],
            category=category,
            privacy=privacy,
            made_for_kids=made_for_kids,
            publish_at=publish_at,
        )
        results.append(res.__dict__)

    successful = sum(1 for r in results if r.get("success"))
    set_upload_job(
        job_id,
        status="completed",
        uploaded=successful,
        progress_pct=100,
        results=results,
    )
