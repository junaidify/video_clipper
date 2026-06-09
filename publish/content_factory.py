"""
Content Factory — Orchestrator Module
Full pipeline: trending topics → script → visuals → TTS → editing → music → final output.
Manages job state, progress tracking, error recovery.
"""

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Import pipeline modules
from ai.trend_scout import scout_trending, get_available_categories
from ai.script_generator import generate_script, get_style_presets, Script
import ai.script_generator as script_generator
from ai.visual_engine import fetch_visuals_for_script
from post.video_editor import assemble_video, EditConfig
from media.music_mixer import (
    search_pixabay_music, get_local_music, download_music,
    mix_background_music, get_mood_options
)

# TTS reuse from existing clipper
try:
    from media.tts_engine import generate_tts
except ImportError:
    generate_tts = None


@dataclass
class FactoryJob:
    """Tracks state of a content generation job."""
    job_id: str
    topic: str
    status: str = "pending"         # pending, scripting, visuals, tts, editing, music, done, failed
    progress: float = 0.0           # 0-100
    message: str = ""
    script: Optional[dict] = None
    visuals: list = field(default_factory=list)
    tts_path: str = ""
    assembled_path: str = ""
    final_path: str = ""
    output_dir: str = ""
    error: str = ""
    created_at: float = 0.0
    completed_at: float = 0.0
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "topic": self.topic,
            "status": self.status,
            "progress": round(self.progress, 1),
            "message": self.message,
            "script": self.script,
            "visuals": self.visuals,
            "tts_path": self.tts_path,
            "assembled_path": self.assembled_path,
            "final_path": self.final_path,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "config": self.config,
        }


# Global job registry
_factory_jobs: dict[str, FactoryJob] = {}
_factory_lock = threading.Lock()


def _update_job(job: FactoryJob, **kwargs):
    """Thread-safe update of job fields."""
    with _factory_lock:
        for k, v in kwargs.items():
            setattr(job, k, v)


def get_job(job_id: str) -> Optional[FactoryJob]:
    with _factory_lock:
        return _factory_jobs.get(job_id)


def list_jobs() -> list[dict]:
    with _factory_lock:
        return [j.to_dict() for j in _factory_jobs.values()]


def start_generation(topic: str,
                     topic_description: str = "",
                     topic_keywords: list = None,
                     target_duration: int = 45,
                     tone: str = "energetic",
                     style: str = "informative",
                     music_mood: str = "upbeat",
                     music_volume: float = 0.15,
                     color_grade: str = "cinematic",
                     output_base_dir: str = None) -> str:
    """
    Start an async content generation job.

    Returns:
        job_id for polling status
    """
    job_id = str(uuid.uuid4())[:8]

    if output_base_dir is None:
        output_base_dir = os.path.join(os.path.dirname(__file__), "factory_output")

    output_dir = os.path.join(output_base_dir, job_id)
    os.makedirs(output_dir, exist_ok=True)

    job = FactoryJob(
        job_id=job_id,
        topic=topic,
        status="pending",
        output_dir=output_dir,
        created_at=time.time(),
        config={
            "target_duration": target_duration,
            "tone": tone,
            "style": style,
            "music_mood": music_mood,
            "music_volume": music_volume,
            "color_grade": color_grade,
            "topic_description": topic_description,
            "topic_keywords": topic_keywords or [],
        }
    )

    with _factory_lock:
        _factory_jobs[job_id] = job

    # Run pipeline in background thread
    thread = threading.Thread(
        target=_run_pipeline, args=(job,), daemon=True
    )
    thread.start()

    return job_id


def _run_pipeline(job: FactoryJob):
    """Execute the full content generation pipeline."""
    cfg = job.config

    try:
        # ── Stage 1: Script Generation (0-20%) ──
        _update_job(job, status="scripting", message="Generating original script...", progress=5)

        script = generate_script(
            topic=job.topic,
            topic_description=cfg.get("topic_description", ""),
            topic_keywords=cfg.get("topic_keywords", []),
            target_duration=cfg.get("target_duration", 45),
            tone=cfg.get("tone", "energetic"),
            style=cfg.get("style", "informative"),
        )

        if not script:
            detail = getattr(script_generator, '_last_error', '') or 'Unknown error'
            _update_job(job, status="failed", error=f"Script generation failed: {detail}")
            logger.error(f"[{job.job_id}] Script failed: {detail}")
            return

        _update_job(job, script=script.to_dict(), progress=20,
                    message=f"Script ready: {len(script.scenes)} scenes")
        logger.info(f"[{job.job_id}] Script done: '{script.title}'")

        # Save script to disk
        script_path = os.path.join(job.output_dir, "script.json")
        with open(script_path, "w") as f:
            json.dump(script.to_dict(), f, indent=2)

        # ── Stage 2: Visual Fetching (20-50%) ──
        _update_job(job, status="visuals", message="Fetching visuals for each scene...", progress=25)

        visuals_dir = os.path.join(job.output_dir, "visuals")
        visual_clips = fetch_visuals_for_script(
            script.to_dict()["scenes"], visuals_dir
        )

        success_count = sum(1 for v in visual_clips if v.success)
        if success_count == 0:
            _update_job(job, status="failed", error="No visuals could be fetched for any scene")
            return

        _update_job(job, visuals=[v.to_dict() for v in visual_clips], progress=50,
                    message=f"Visuals ready: {success_count}/{len(visual_clips)} scenes")
        logger.info(f"[{job.job_id}] Visuals: {success_count}/{len(visual_clips)}")

        # ── Stage 3: TTS Narration (50-65%) ──
        _update_job(job, status="tts", message="Generating voiceover narration...", progress=55)

        # Combine all scene narrations into full script
        full_narration = " ".join(s.narration for s in script.scenes)
        tts_path = os.path.join(job.output_dir, "narration.mp3")

        tts_success = False
        if generate_tts:
            try:
                tts_result = generate_tts(
                    text=full_narration,
                    output_path=tts_path,
                    voice="en-US-Neural2-D",  # deep male voice
                    speed=1.05,
                )
                tts_success = isinstance(tts_result, dict) and tts_result.get("success", False)
                if not tts_success and isinstance(tts_result, str) and os.path.isfile(tts_result):
                    tts_success = True
                    tts_path = tts_result
            except Exception as e:
                logger.warning(f"TTS failed: {e}")

        if not tts_success:
            # Fallback: generate silent audio placeholder
            logger.warning(f"[{job.job_id}] TTS failed, using silent placeholder")
            from media.music_mixer import generate_silent_tone
            generate_silent_tone(script.total_duration, tts_path)

        _update_job(job, tts_path=tts_path, progress=65, message="Voiceover ready")

        # ── Stage 4: Video Assembly (65-85%) ──
        _update_job(job, status="editing", message="Assembling and editing video...", progress=70)

        # Build scene clip list for the editor
        scene_clip_data = []
        for i, scene in enumerate(script.scenes):
            visual = visual_clips[i] if i < len(visual_clips) else None
            if visual and visual.success:
                scene_clip_data.append({
                    "clip_path": visual.clip_path,
                    "duration": scene.end_time - scene.start_time,
                    "transition": scene.transition,
                    "text_overlay": scene.text_overlay,
                    "scene_number": scene.scene_number,
                })

        assembled_path = os.path.join(job.output_dir, "assembled.mp4")

        edit_config = EditConfig(
            color_correction=True,
            ken_burns=True,
            text_overlays=True,
            output_quality=20,
        )

        edit_result = assemble_video(scene_clip_data, assembled_path, edit_config)

        if not edit_result.success:
            _update_job(job, status="failed", error=f"Video assembly failed: {edit_result.error}")
            return

        _update_job(job, assembled_path=assembled_path, progress=80,
                    message="Video assembled, adding audio...")

        # ── Stage 4b: Mux TTS audio onto assembled video ──
        narrated_path = os.path.join(job.output_dir, "narrated.mp4")
        _mux_audio(assembled_path, tts_path, narrated_path)

        if not os.path.isfile(narrated_path):
            # Fallback: use assembled without narration
            narrated_path = assembled_path

        _update_job(job, progress=85, message="Narration synced")

        # ── Stage 5: Background Music (85-95%) ──
        _update_job(job, status="music", message="Adding background music...", progress=88)

        music_mood = cfg.get("music_mood", "upbeat")
        music_vol = cfg.get("music_volume", 0.15)
        final_path = os.path.join(job.output_dir, "final.mp4")

        music_added = False
        # Try to find music
        tracks = search_pixabay_music(music_mood)
        if not tracks:
            tracks = get_local_music(music_mood)

        if tracks:
            music_file = os.path.join(job.output_dir, "bg_music.mp3")
            if download_music(tracks[0], music_file):
                mix_result = mix_background_music(
                    narrated_path, music_file, final_path,
                    music_volume=music_vol,
                    duck_enabled=True,
                )
                music_added = mix_result.success

        if not music_added:
            # No music available — final = narrated
            shutil.copy2(narrated_path, final_path)
            logger.info(f"[{job.job_id}] No background music — skipped")

        _update_job(job, progress=95, message="Final touches...")

        # ── Stage 6: Finalize (95-100%) ──
        if os.path.isfile(final_path):
            _update_job(job, final_path=final_path, status="done", progress=100,
                        message="Video ready!", completed_at=time.time())
            elapsed = job.completed_at - job.created_at
            logger.info(
                f"[{job.job_id}] Content factory complete: "
                f"'{script.title}' in {elapsed:.0f}s"
            )
        else:
            _update_job(job, status="failed", error="Final output file missing")

    except Exception as e:
        logger.error(f"[{job.job_id}] Pipeline error: {e}")
        _update_job(job, status="failed", error=str(e))


def _mux_audio(video_path: str, audio_path: str, output_path: str) -> bool:
    """Mux audio onto a video (replacing any existing audio)."""
    import subprocess

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=120)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Audio mux failed: {e}")
        return False


def cleanup_job(job_id: str) -> bool:
    """Remove a job and its output directory."""
    with _factory_lock:
        job = _factory_jobs.pop(job_id, None)
    if job and job.output_dir and os.path.isdir(job.output_dir):
        shutil.rmtree(job.output_dir, ignore_errors=True)
        return True
    return False
