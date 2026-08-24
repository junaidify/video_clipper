"""
Content Factory — Pipeline Orchestrator Module
End-to-end autonomous content creation pipeline:
Trending topics / Custom story -> Script -> Visuals -> TTS -> Multi-scene Assembly -> Music -> Final Export.
"""
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict

from video_clipper.content_generation.trend_scout import scout_trending, get_available_categories
from video_clipper.content_generation.script_generator import generate_script, get_style_presets, Script, Scene
import video_clipper.content_generation.script_generator as script_generator
from video_clipper.content_generation.visual_engine import fetch_visuals_for_script, image_to_video, generate_color_bg, VisualClip
from video_clipper.video_processing.video_editor import assemble_video, EditConfig, _get_duration
from video_clipper.audio_processing.music_mixer import (
    search_pixabay_music, get_local_music, download_music,
    mix_background_music, get_mood_options, generate_silent_tone
)

logger = logging.getLogger(__name__)


@dataclass
class FactoryJob:
    """State and progress tracker for an automated content generation job."""
    job_id: str
    topic: str
    mode: str = "auto"              # "auto" or "custom"
    status: str = "pending"         # pending, scripting, visuals, tts, editing, music, done, failed
    progress: float = 0.0           # 0-100
    message: str = ""
    script: Optional[dict] = None
    visuals: List[dict] = field(default_factory=list)
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
            "mode": self.mode,
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
_factory_jobs: Dict[str, FactoryJob] = {}
_factory_lock = threading.Lock()


def _update_job(job: FactoryJob, **kwargs):
    with _factory_lock:
        for k, v in kwargs.items():
            setattr(job, k, v)


def get_job(job_id: str) -> Optional[FactoryJob]:
    with _factory_lock:
        return _factory_jobs.get(job_id)


def list_jobs() -> List[dict]:
    with _factory_lock:
        return [j.to_dict() for j in _factory_jobs.values()]


def start_generation(
    topic: str,
    topic_description: str = "",
    topic_keywords: Optional[List[str]] = None,
    target_duration: int = 45,
    tone: str = "energetic",
    style: str = "informative",
    music_mood: str = "upbeat",
    music_volume: float = 0.15,
    color_grade: str = "cinematic",
    video_format: str = "9:16",
    output_base_dir: Optional[str] = None,
) -> str:
    """Start an async automated content generation job."""
    job_id = str(uuid.uuid4())[:8]

    if output_base_dir is None:
        output_base_dir = os.path.join(os.getcwd(), "factory_output")

    output_dir = os.path.join(output_base_dir, job_id)
    os.makedirs(output_dir, exist_ok=True)

    job = FactoryJob(
        job_id=job_id,
        topic=topic,
        mode="auto",
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
            "video_format": video_format,
            "topic_description": topic_description,
            "topic_keywords": topic_keywords or [],
        }
    )

    with _factory_lock:
        _factory_jobs[job_id] = job

    thread = threading.Thread(target=_run_pipeline, args=(job,), daemon=True)
    thread.start()

    return job_id


def start_custom_generation(
    job_id: str,
    title: str,
    script_text: str,
    image_paths: List[str],
    music_mood: str = "upbeat",
    music_volume: float = 0.15,
    video_format: str = "9:16",
    output_base_dir: Optional[str] = None,
) -> str:
    """Start an async custom content generation job using provided script and user assets."""
    if output_base_dir is None:
        output_base_dir = os.path.join(os.getcwd(), "factory_output")

    output_dir = os.path.join(output_base_dir, job_id)
    os.makedirs(output_dir, exist_ok=True)

    job = FactoryJob(
        job_id=job_id,
        topic=title,
        mode="custom",
        status="pending",
        output_dir=output_dir,
        created_at=time.time(),
        config={
            "title": title,
            "script_text": script_text,
            "image_paths": image_paths,
            "music_mood": music_mood,
            "music_volume": music_volume,
            "video_format": video_format,
        }
    )

    with _factory_lock:
        _factory_jobs[job_id] = job

    thread = threading.Thread(target=_run_pipeline, args=(job,), daemon=True)
    thread.start()

    return job_id


def _run_pipeline(job: FactoryJob):
    """Execute the multi-stage content generation pipeline."""
    cfg = job.config

    try:
        # Stage 1: Script Generation
        _update_job(job, status="scripting", message="Generating script...", progress=5)

        if job.mode == "custom":
            script_text = cfg.get("script_text", "")
            title = cfg.get("title", "Custom Story")
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +|\n+', script_text) if s.strip()]
            if not sentences:
                sentences = ["No script provided."]

            script = Script(
                title=title,
                topic=title,
                hook=sentences[0] if sentences else "",
                scenes=[]
            )
            for i, text in enumerate(sentences):
                script.scenes.append(Scene(
                    scene_number=i + 1,
                    start_time=0.0,
                    end_time=0.0,
                    narration=text,
                    visual_description="User provided image",
                    visual_keywords=[],
                    visual_type="user_image",
                    text_overlay=text,
                    transition="xfade" if i > 0 else "fade"
                ))

            _update_job(job, script=script.to_dict(), progress=20,
                        message=f"Custom script ready: {len(script.scenes)} scenes")
        else:
            script = generate_script(
                topic=job.topic,
                topic_description=cfg.get("topic_description", ""),
                topic_keywords=cfg.get("topic_keywords", []),
                target_duration=cfg.get("target_duration", 45),
                tone=cfg.get("tone", "energetic"),
                style=cfg.get("style", "informative"),
            )

            if not script:
                detail = getattr(script_generator, '_last_error', '') or 'LLM script generation failed'
                _update_job(job, status="failed", error=f"Script generation failed: {detail}")
                return

            _update_job(job, script=script.to_dict(), progress=20,
                        message=f"Script ready: {len(script.scenes)} scenes")

        script_path = os.path.join(job.output_dir, "script.json")
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(script.to_dict(), f, indent=2)

        # Stage 2: TTS Voiceover
        _update_job(job, status="tts", message="Generating voiceover narration...", progress=25)
        full_narration = " ".join(s.narration for s in script.scenes)
        tts_path = os.path.join(job.output_dir, "narration.mp3")

        tts_success = False
        try:
            from video_clipper.audio_processing.tts_engine import _run_async, _synthesize_segment
            result = _run_async(_synthesize_segment(
                text=full_narration,
                voice_id="en-US-ChristopherNeural",
                output_path=tts_path
            ))
            tts_success = bool(result and result.get("path"))
        except Exception as e:
            logger.warning(f"TTS failed: {e}")

        if not tts_success:
            generate_silent_tone(script.total_duration if job.mode != "custom" else 30.0, tts_path)

        _update_job(job, tts_path=tts_path, progress=40, message="Voiceover ready")

        video_format = cfg.get("video_format", "9:16")
        is_landscape = (video_format == "16:9")
        target_w = 1920 if is_landscape else 1080
        target_h = 1080 if is_landscape else 1920

        total_audio_dur = _get_duration(tts_path) if os.path.isfile(tts_path) else (
            script.total_duration if job.mode != "custom" else 30.0
        )

        if job.mode == "custom":
            total_chars = sum(len(s.narration) for s in script.scenes) or 1
            current_time = 0.0
            for s in script.scenes:
                s_dur = max((len(s.narration) / total_chars) * total_audio_dur, 2.0)
                s.start_time = current_time
                s.end_time = current_time + s_dur
                current_time += s_dur

        # Stage 3: Visual Assets
        _update_job(job, status="visuals", message="Preparing visuals for each scene...", progress=45)
        visuals_dir = os.path.join(job.output_dir, "visuals")
        os.makedirs(visuals_dir, exist_ok=True)

        if job.mode == "custom":
            image_paths = cfg.get("image_paths", [])
            visual_clips = []
            for i, scene in enumerate(script.scenes):
                img_path = image_paths[i % len(image_paths)] if image_paths else ""
                clip_path = os.path.join(visuals_dir, f"scene_{i+1:02d}.mp4")
                duration = scene.end_time - scene.start_time

                success = False
                if img_path and os.path.isfile(img_path):
                    success = image_to_video(img_path, duration, clip_path, width=target_w, height=target_h)

                if not success:
                    success = generate_color_bg(duration, clip_path, width=target_w, height=target_h)

                visual_clips.append(VisualClip(
                    scene_number=i + 1,
                    clip_path=clip_path,
                    duration=duration,
                    visual_type="user_image",
                    source="user",
                    success=success,
                ))
        else:
            visual_clips = fetch_visuals_for_script(
                script.to_dict()["scenes"], visuals_dir, video_format=video_format
            )

        success_count = sum(1 for v in visual_clips if v.success)
        if success_count == 0:
            _update_job(job, status="failed", error="No visual scenes could be generated")
            return

        _update_job(job, visuals=[v.to_dict() for v in visual_clips], progress=65,
                    message=f"Visuals ready: {success_count}/{len(visual_clips)} scenes")

        # Stage 4: Video Assembly
        _update_job(job, status="editing", message="Assembling and editing video...", progress=70)

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
            width=target_w,
            height=target_h,
        )

        edit_result = assemble_video(scene_clip_data, assembled_path, edit_config)
        if not edit_result.success:
            _update_job(job, status="failed", error=f"Video assembly failed: {edit_result.error}")
            return

        _update_job(job, assembled_path=assembled_path, progress=80,
                    message="Video assembled, adding audio...")

        # Stage 4b: Mux TTS voiceover
        narrated_path = os.path.join(job.output_dir, "narrated.mp4")
        _mux_audio(assembled_path, tts_path, narrated_path)

        if not os.path.isfile(narrated_path):
            narrated_path = assembled_path

        _update_job(job, progress=85, message="Narration synced")

        # Stage 5: Background Music
        _update_job(job, status="music", message="Adding background music...", progress=88)

        music_mood = cfg.get("music_mood", "upbeat")
        music_vol = cfg.get("music_volume", 0.15)
        final_path = os.path.join(job.output_dir, "final.mp4")

        music_added = False
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
            shutil.copy2(narrated_path, final_path)

        _update_job(job, progress=95, message="Final touches...")

        # Stage 6: Completion
        if os.path.isfile(final_path):
            _update_job(job, final_path=final_path, status="done", progress=100,
                        message="Video ready!", completed_at=time.time())
            elapsed = job.completed_at - job.created_at
            logger.info(f"[{job.job_id}] Content factory complete in {elapsed:.0f}s: '{script.title}'")
        else:
            _update_job(job, status="failed", error="Final output file missing")

    except Exception as e:
        logger.error(f"[{job.job_id}] Pipeline error: {e}")
        _update_job(job, status="failed", error=str(e))


def _mux_audio(video_path: str, audio_path: str, output_path: str) -> bool:
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
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=120)
        return result.returncode == 0
    except Exception:
        return False


def cleanup_job(job_id: str) -> bool:
    with _factory_lock:
        job = _factory_jobs.pop(job_id, None)
    if job and job.output_dir and os.path.isdir(job.output_dir):
        shutil.rmtree(job.output_dir, ignore_errors=True)
        return True
    return False
