"""
Full Video Processor Module
Applies full short-form / vertical production pipeline to full-length videos of any duration:
1. 9:16 vertical crop with blurred background fill
2. Word-level subtitle generation + burning
3. Visual text overlay
4. Anti-copyright video modulation
5. YouTube-style thumbnail generation
"""
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class FullVideoConfig:
    """Configuration for full video processing."""
    crop_vertical: bool = True
    vertical_width: int = 1080
    vertical_height: int = 1920
    video_quality: int = 23

    # Subtitles
    enable_subtitles: bool = False
    subtitle_model_size: str = "base"
    subtitle_language: Optional[str] = None

    # Text overlay
    enable_overlay: bool = False
    overlay_text: str = ""
    overlay_position_x: float = 0.5
    overlay_position_y: float = 0.15
    overlay_font_size: int = 48
    overlay_bg_opacity: float = 0.6

    # Thumbnail
    enable_thumbnail: bool = False
    thumbnail_title: str = ""
    thumbnail_style: str = "bold"

    # Modulation (hash-breaking)
    enable_modulation: bool = False
    modulation_preset: str = ""
    modulation_config: dict = field(default_factory=dict)


@dataclass
class FullVideoResult:
    """Result of full video processing pipeline."""
    success: bool
    output_path: str = ""
    thumbnail_path: str = ""
    srt_path: str = ""
    error: Optional[str] = None
    processing_steps: List[str] = field(default_factory=list)
    duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_path": self.output_path,
            "thumbnail_path": self.thumbnail_path,
            "srt_path": self.srt_path,
            "error": self.error,
            "processing_steps": self.processing_steps,
            "duration": self.duration,
        }


def _get_video_duration(video_path: str) -> float:
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
               "-show_format", video_path]
        r = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        info = json.loads(r.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0


def _get_video_dimensions(video_path: str) -> tuple:
    try:
        cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
               "-show_entries", "stream=width,height", "-of", "json", video_path]
        r = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        data = json.loads(r.stdout)
        s = data["streams"][0]
        return int(s["width"]), int(s["height"])
    except Exception:
        return 1920, 1080


def _crop_to_vertical(video_path: str, output_path: str, config: FullVideoConfig) -> bool:
    """Apply 9:16 scale-to-fit with blurred background fill."""
    tw, th = config.vertical_width, config.vertical_height

    filter_complex = (
        f"[0:v]split=2[bg_in][fg_in];"
        f"[bg_in]scale={tw}:{th}:force_original_aspect_ratio=increase,"
        f"crop={tw}:{th},gblur=sigma=60,"
        f"eq=brightness=-0.12:saturation=1.3,vignette=PI/3.5[bg];"
        f"[fg_in]scale={tw}:{th}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2:format=auto[vout]"
    )

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-threads", "0",
        "-crf", str(config.video_quality),
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        "-movflags", "+faststart",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                            errors='replace', timeout=600)
    if result.returncode != 0:
        logger.error(f"Vertical crop failed: {result.stderr[-300:]}")
        return False
    return os.path.isfile(output_path) and os.path.getsize(output_path) > 1000


def process_full_video(
    video_path: str,
    output_dir: str,
    config: Optional[FullVideoConfig] = None,
    progress_callback=None,
) -> FullVideoResult:
    """
    Process a full-length video with all requested post-production features.

    Pipeline:
    1. 9:16 vertical crop
    2. Subtitle generation & burn-in
    3. Text overlay
    4. Video modulation
    5. Finalizing output
    6. Thumbnail generation
    """
    config = config or FullVideoConfig()
    os.makedirs(output_dir, exist_ok=True)

    steps = []
    stem = Path(video_path).stem
    working_path = video_path

    def _update(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    try:
        duration = _get_video_duration(video_path)

        # Step 1: Vertical Crop
        if config.crop_vertical:
            _update(10, "Applying 9:16 vertical crop...")
            cropped_path = os.path.join(output_dir, f"{stem}_vertical.mp4")
            if _crop_to_vertical(working_path, cropped_path, config):
                working_path = cropped_path
                steps.append("vertical_crop")
                logger.info("Step 1: Vertical crop complete.")
            else:
                logger.warning("Vertical crop failed, continuing with source file.")

        # Step 2: Subtitles
        srt_path = ""
        if config.enable_subtitles:
            _update(30, "Generating subtitles...")
            try:
                from video_clipper.video_processing.subtitle_generator import generate_subtitles, burn_subtitles

                sub_result = generate_subtitles(
                    working_path,
                    model_size=config.subtitle_model_size,
                    language=config.subtitle_language,
                )

                srt_path = os.path.join(output_dir, f"{stem}.srt")
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write(sub_result['srt'])

                sub_out = os.path.join(output_dir, f"{stem}_subtitled.mp4")
                burn_subtitles(working_path, sub_result['srt'], sub_out)
                working_path = sub_out
                steps.append("subtitles")
                logger.info(f"Step 2: Subtitles burned ({sub_result['method']}).")
            except Exception as e:
                logger.warning(f"Subtitle step failed: {e}")

        # Step 3: Text Overlay
        if config.enable_overlay and config.overlay_text:
            _update(50, "Burning text overlay...")
            try:
                from video_clipper.video_processing.subtitle_generator import burn_text_overlay

                overlay_out = os.path.join(output_dir, f"{stem}_overlay.mp4")
                text_blocks = [{
                    'words': [
                        {'word': w, 'color': '#CAFF00' if i == 0 else '#FFFFFF'}
                        for i, w in enumerate(config.overlay_text.split())
                    ],
                    'x': config.overlay_position_x,
                    'y': config.overlay_position_y,
                    'font_size': config.overlay_font_size,
                    'bg_opacity': config.overlay_bg_opacity,
                }]
                burn_text_overlay(working_path, overlay_out, text_blocks)
                working_path = overlay_out
                steps.append("text_overlay")
                logger.info("Step 3: Text overlay complete.")
            except Exception as e:
                logger.warning(f"Text overlay step failed: {e}")

        # Step 4: Modulation
        if config.enable_modulation:
            _update(65, "Applying modulation transforms...")
            try:
                from video_clipper.video_processing.video_modulator import modulate_video, ModulationConfig, get_presets

                if config.modulation_preset:
                    presets = get_presets()
                    if config.modulation_preset in presets:
                        mod_cfg = ModulationConfig(**presets[config.modulation_preset]["config"])
                    else:
                        mod_cfg = ModulationConfig()
                elif config.modulation_config:
                    mod_cfg = ModulationConfig(**config.modulation_config)
                else:
                    mod_cfg = ModulationConfig()

                mod_out = os.path.join(output_dir, f"{stem}_mod.mp4")
                mod_result = modulate_video(working_path, mod_out, mod_cfg)
                if mod_result.success:
                    working_path = mod_out
                    steps.append(f"modulation:{','.join(mod_result.transforms_applied)}")
                    logger.info(f"Step 4: Modulation done ({mod_result.transforms_applied}).")
                else:
                    logger.warning(f"Modulation failed: {mod_result.error}")
            except Exception as e:
                logger.warning(f"Modulation step failed: {e}")

        # Step 5: Finalize output name
        _update(85, "Finalizing output...")
        final_name = f"{stem}_processed.mp4"
        final_path = os.path.join(output_dir, final_name)
        if working_path != final_path:
            if working_path != video_path:
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(working_path, final_path)
            else:
                import shutil
                shutil.copy2(working_path, final_path)

        # Step 6: Thumbnail
        thumbnail_path = ""
        if config.enable_thumbnail:
            _update(90, "Generating thumbnail...")
            try:
                from video_clipper.video_processing.thumbnail_generator import generate_template_thumbnail
                title = config.thumbnail_title or stem
                thumbnail_path = os.path.join(output_dir, f"{stem}_thumb.png")
                generate_template_thumbnail(
                    final_path, title, thumbnail_path, style=config.thumbnail_style
                )
                steps.append("thumbnail")
                logger.info("Step 6: Thumbnail generated.")
            except Exception as e:
                logger.warning(f"Thumbnail step failed: {e}")

        _update(100, "Complete")

        return FullVideoResult(
            success=True,
            output_path=final_path,
            thumbnail_path=thumbnail_path,
            srt_path=srt_path,
            processing_steps=steps,
            duration=duration,
        )

    except Exception as e:
        logger.exception("Full video processing failed")
        return FullVideoResult(
            success=False,
            error=str(e),
            processing_steps=steps,
        )
