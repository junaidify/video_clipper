"""
Content Generation module: automated trend scouting, script writing, commentary, visual synthesis, and full factory orchestration.
"""
from video_clipper.content_generation.trend_scout import (
    TrendingTopic,
    fetch_youtube_trending,
    fetch_google_trends,
    fetch_reddit_trending,
    scout_trending,
    get_available_categories,
)
from video_clipper.content_generation.script_generator import (
    Scene,
    Script,
    generate_script,
    get_style_presets,
)
from video_clipper.content_generation.commentary import (
    CommentarySegment,
    CommentaryScript,
    generate_commentary,
)
from video_clipper.content_generation.visual_engine import (
    VisualClip,
    fetch_visuals_for_script,
    image_to_video,
    generate_color_bg,
)
from video_clipper.content_generation.factory_orchestrator import (
    FactoryJob,
    start_generation,
    start_custom_generation,
    get_job,
    list_jobs,
    cleanup_job,
)

__all__ = [
    "TrendingTopic",
    "fetch_youtube_trending",
    "fetch_google_trends",
    "fetch_reddit_trending",
    "scout_trending",
    "get_available_categories",
    "Scene",
    "Script",
    "generate_script",
    "get_style_presets",
    "CommentarySegment",
    "CommentaryScript",
    "generate_commentary",
    "VisualClip",
    "fetch_visuals_for_script",
    "image_to_video",
    "generate_color_bg",
    "FactoryJob",
    "start_generation",
    "start_custom_generation",
    "get_job",
    "list_jobs",
    "cleanup_job",
]
