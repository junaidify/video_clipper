"""
Video Processing module: scene assembly, subtitles, thumbnails, and modulation.
"""
from video_clipper.video_processing.video_editor import (
    EditConfig,
    EditResult,
    assemble_video,
    add_text_overlay,
    apply_color_grade,
)
from video_clipper.video_processing.subtitle_generator import (
    generate_subtitles,
    burn_subtitles,
    burn_text_overlay,
    words_to_srt,
)
from video_clipper.video_processing.thumbnail_generator import (
    score_frame,
    pick_top_frames,
    extract_frames,
    pick_best_frame,
    generate_template_thumbnail,
    generate_ai_thumbnail,
)
from video_clipper.video_processing.video_modulator import (
    ModulationConfig,
    ModulationResult,
    modulate_video,
    get_presets,
)

__all__ = [
    "EditConfig",
    "EditResult",
    "assemble_video",
    "add_text_overlay",
    "apply_color_grade",
    "generate_subtitles",
    "burn_subtitles",
    "burn_text_overlay",
    "words_to_srt",
    "score_frame",
    "pick_top_frames",
    "extract_frames",
    "pick_best_frame",
    "generate_template_thumbnail",
    "generate_ai_thumbnail",
    "ModulationConfig",
    "ModulationResult",
    "modulate_video",
    "get_presets",
]
