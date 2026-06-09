"""
Sequential Splitter Module
Splits an entire video into consecutive reels/shorts of target duration.
No frame is skipped — the full video is covered end to end.

Features:
- Sentence-boundary aware splitting (uses transcript)
- 1-2 second overlap between consecutive reels for continuity
- Configurable target reel duration (default 50-60s)
- 9:16 fit with blurred background
"""
import json
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SequentialConfig:
    """Configuration for sequential splitting."""
    target_duration: int = 55       # target reel length in seconds
    min_duration: int = 30          # won't create reels shorter than this
    max_duration: int = 65          # hard cap
    overlap_seconds: float = 1.5    # overlap between consecutive reels
    crop_vertical: bool = True
    vertical_width: int = 1080
    vertical_height: int = 1920
    video_quality: int = 23


def find_sentence_boundaries(transcript, target_times: list,
                             search_window: float = 5.0) -> list:
    """
    Given a list of target split times, snap each to the nearest
    sentence boundary from the transcript.

    Args:
        transcript: Transcript object with segments
        target_times: List of desired split times (seconds)
        search_window: How many seconds to search around each target

    Returns:
        List of adjusted split times snapped to sentence ends
    """
    if not transcript or not transcript.segments:
        return target_times

    # Collect all sentence end times
    sentence_ends = [seg.end for seg in transcript.segments]

    adjusted = []
    for target in target_times:
        # Find the closest sentence end within the search window
        best = target
        best_dist = search_window + 1

        for end_time in sentence_ends:
            dist = abs(end_time - target)
            if dist < best_dist and dist <= search_window:
                best = end_time
                best_dist = dist

        adjusted.append(round(best, 2))

    return adjusted


def compute_split_points(video_duration: float, config: SequentialConfig,
                         transcript=None) -> list:
    """
    Compute split points for sequential reeling.

    Returns list of (start, end) tuples covering the entire video.
    Each reel overlaps the next by config.overlap_seconds.
    """
    if video_duration <= 0:
        return []

    target = config.target_duration
    overlap = config.overlap_seconds

    # Calculate raw split points (end times for each reel)
    raw_points = []
    current = 0.0
    while current < video_duration:
        end = min(current + target, video_duration)
        raw_points.append(end)

        if end >= video_duration:
            break  # reached the end, no more reels

        # Next reel starts overlap_seconds before this one ends
        next_start = end - overlap
        # Safety: always advance by at least 1 second
        if next_start <= current:
            next_start = current + max(target - overlap, 1.0)
        current = next_start

    # Snap to sentence boundaries if transcript is available
    if transcript:
        raw_points = find_sentence_boundaries(transcript, raw_points)

    # Build (start, end) pairs
    splits = []
    prev_end = 0.0
    for i, end_time in enumerate(raw_points):
        if i == 0:
            start = 0.0
        else:
            # Start with overlap from previous reel
            start = max(0, prev_end - overlap)

        # Ensure we don't exceed video
        end_time = min(end_time, video_duration)

        # Skip tiny remnants
        if end_time - start < config.min_duration and i > 0:
            # Extend the previous reel instead
            if splits:
                prev_start, _ = splits[-1]
                splits[-1] = (prev_start, end_time)
            prev_end = end_time
            continue

        # Enforce max duration
        if end_time - start > config.max_duration:
            end_time = start + config.max_duration

        splits.append((round(start, 2), round(end_time, 2)))
        prev_end = end_time

    # Make sure last reel covers to the end
    if splits and splits[-1][1] < video_duration - 2:
        last_start = max(0, video_duration - target)
        if last_start < splits[-1][1]:
            # Extend last reel
            s, _ = splits[-1]
            splits[-1] = (s, round(video_duration, 2))
        else:
            splits.append((round(last_start, 2), round(video_duration, 2)))

    return splits


def split_sequentially(video_path: str, output_dir: str,
                       config: Optional[SequentialConfig] = None,
                       transcript=None,
                       video_duration: Optional[float] = None) -> list:
    """
    Split entire video into sequential reels.

    Args:
        video_path: Path to source video
        output_dir: Output directory for reels
        config: Sequential splitting configuration
        transcript: Optional transcript for sentence-boundary snapping
        video_duration: Video duration (auto-detected if not provided)

    Returns:
        List of result dicts per reel
    """
    config = config or SequentialConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video duration if not provided
    if not video_duration:
        video_duration = _get_duration(video_path)

    if video_duration <= 0:
        logger.error("Could not determine video duration")
        return []

    # Compute split points
    splits = compute_split_points(video_duration, config, transcript)
    total_reels = len(splits)

    logger.info(
        f"Sequential split: {video_duration:.0f}s video → "
        f"{total_reels} reels (~{config.target_duration}s each, "
        f"{config.overlap_seconds}s overlap)"
    )

    # Get video dimensions for vertical fit
    src_w, src_h = _get_dimensions(video_path)

    results = []
    for i, (start, end) in enumerate(splits, 1):
        duration = end - start
        output_name = f"reel_{i:03d}_of_{total_reels}_{start:.0f}s-{end:.0f}s.mp4"
        output_path = str(output_dir / output_name)

        logger.info(f"  Reel {i}/{total_reels}: {start:.1f}s - {end:.1f}s ({duration:.1f}s)")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-avoid_negative_ts", "1",
        ]

        # 9:16 fit with blurred background
        if config.crop_vertical and src_w and src_h:
            tw, th = config.vertical_width, config.vertical_height
            filter_complex = (
                f"[0:v]split=2[bg_in][fg_in];"
                f"[bg_in]scale={tw}:{th}:force_original_aspect_ratio=increase,"
                f"crop={tw}:{th},gblur=sigma=60,"
                f"eq=brightness=-0.12:saturation=1.3,vignette=PI/3.5[bg];"
                f"[fg_in]scale={tw}:{th}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2:format=auto[vout]"
            )
            cmd.extend(["-filter_complex", filter_complex])
            cmd.extend(["-map", "[vout]", "-map", "0:a?"])

        cmd.extend([
            "-c:v", "libx264", "-preset", "veryfast", "-threads", "0",
            "-crf", str(config.video_quality),
            "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            "-movflags", "+faststart",
            output_path
        ])

        try:
            result = subprocess.run(
                cmd, capture_output=True, encoding='utf-8',
                errors='replace', timeout=180
            )

            if result.returncode != 0:
                error = result.stderr[-300:] if result.stderr else "Unknown"
                results.append({
                    "reel_number": i,
                    "total_reels": total_reels,
                    "filename": output_name,
                    "start": start, "end": end,
                    "duration": duration,
                    "success": False,
                    "error": error,
                })
            else:
                out_file = Path(output_path)
                size_mb = out_file.stat().st_size / (1024*1024) if out_file.exists() else 0
                results.append({
                    "reel_number": i,
                    "total_reels": total_reels,
                    "filename": output_name,
                    "output_path": output_path,
                    "start": start, "end": end,
                    "duration": duration,
                    "size_mb": round(size_mb, 2),
                    "success": True,
                })

        except subprocess.TimeoutExpired:
            results.append({
                "reel_number": i,
                "total_reels": total_reels,
                "filename": output_name,
                "start": start, "end": end,
                "duration": duration,
                "success": False,
                "error": "FFmpeg timed out",
            })

    success = sum(1 for r in results if r.get("success"))
    logger.info(f"Sequential split complete: {success}/{total_reels} reels created")
    return results


def _get_duration(video_path: str) -> float:
    """Get video duration via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", video_path
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        info = json.loads(r.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0


def _get_dimensions(video_path: str) -> tuple:
    """Get video width and height."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", video_path
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        info = json.loads(r.stdout)
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                return int(s["width"]), int(s["height"])
    except Exception:
        pass
    return None, None
