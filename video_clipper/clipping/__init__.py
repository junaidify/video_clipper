"""
Clipping module: intelligent video analysis, speech-to-text, and clip extraction engines.
"""
from video_clipper.clipping.transcriber import (
    TranscriptSegment,
    Transcript,
    transcribe,
    extract_audio,
    resolve_device,
)
from video_clipper.clipping.analyzer import (
    ClipCandidate,
    ContentAnalyzer,
)
from video_clipper.clipping.patterns import (
    PatternMatch,
    StructuralPattern,
    PatternScorer,
)
from video_clipper.clipping.llm_analyzer import (
    LLMConfig,
    analyze_with_llm,
)
from video_clipper.clipping.clipper import (
    ClipResult,
    VideoClipper,
)
from video_clipper.clipping.manual_splitter import (
    TimestampClip,
    parse_timestamp,
    split_by_timestamps,
)
from video_clipper.clipping.sequential_splitter import (
    SequentialConfig,
    compute_split_points,
    find_sentence_boundaries,
    split_sequentially,
)
from video_clipper.clipping.engagement_analyzer import (
    EngagementConfig,
    EngagementSegment,
    EngagementAnalyzer,
    analyze_with_llm_for_engagement,
)
from video_clipper.clipping.full_video_processor import (
    FullVideoConfig,
    FullVideoResult,
    process_full_video,
)

__all__ = [
    "TranscriptSegment",
    "Transcript",
    "transcribe",
    "extract_audio",
    "resolve_device",
    "ClipCandidate",
    "ContentAnalyzer",
    "PatternMatch",
    "StructuralPattern",
    "PatternScorer",
    "LLMConfig",
    "analyze_with_llm",
    "ClipResult",
    "VideoClipper",
    "TimestampClip",
    "parse_timestamp",
    "split_by_timestamps",
    "SequentialConfig",
    "compute_split_points",
    "find_sentence_boundaries",
    "split_sequentially",
    "EngagementConfig",
    "EngagementSegment",
    "EngagementAnalyzer",
    "analyze_with_llm_for_engagement",
    "FullVideoConfig",
    "FullVideoResult",
    "process_full_video",
]
