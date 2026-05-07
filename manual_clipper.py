"""
Manual Timestamp Splitter
Enter start/end timestamps → get a clip. No analysis, no AI.
Supports multiple timestamp pairs in one go.
"""
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TimestampClip:
    """A manually defined clip."""
    start: float    # seconds
    end: float      # seconds
    label: Optional[str] = None  # optional user label

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_timestamp(ts: str) -> float:
    """
    Parse timestamp string to seconds.
    Supports: "90", "1:30", "01:30", "1:30:00", "1h30m", "90s"
    """
    ts = ts.strip()

    # Pure number (seconds)
    try:
        return float(ts)
    except ValueError:
        pass

    # HH:MM:SS or MM:SS format
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)

    # Human format: 1h30m, 2m30s, 90s
    import re
    h = re.search(r'(\d+)\s*h', ts)
    m = re.search(r'(\d+)\s*m', ts)
    s = re.search(r'(\d+\.?\d*)\s*s', ts)

    total = 0
    if h: total += int(h.group(1)) * 3600
    if m: total += int(m.group(1)) * 60
    if s: total += float(s.group(1))

    if total > 0:
        return total

    raise ValueError(f"Cannot parse timestamp: '{ts}'")


def split_by_timestamps(video_path: str, clips: list,
                        output_dir: str, crop_vertical: bool = True,
                        vertical_width: int = 1080,
                        vertical_height: int = 1920) -> list:
    """
    Split video at exact user-defined timestamps.

    Args:
        video_path: Path to source video
        clips: List of TimestampClip objects
        output_dir: Where to save output clips
        crop_vertical: Whether to fit to 9:16
        vertical_width/height: Target dimensions

    Returns:
        List of dicts with result info per clip
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, clip in enumerate(clips, 1):
        if clip.duration <= 0:
            results.append({
                "clip_number": i,
                "success": False,
                "error": f"Invalid duration: start={clip.start}, end={clip.end}"
            })
            continue

        label = clip.label or f"clip_{i:02d}"
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        output_name = f"{safe_label}_{clip.start:.0f}s-{clip.end:.0f}s.mp4"
        output_path = str(output_dir / output_name)

        logger.info(f"Manual split [{i}]: {clip.start:.1f}s - {clip.end:.1f}s ({clip.duration:.1f}s)")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip.start),
            "-i", video_path,
            "-t", str(clip.duration),
            "-avoid_negative_ts", "1",
        ]

        # 9:16 fit with blurred background
        if crop_vertical:
            try:
                # Get dimensions
                probe_cmd = [
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_streams", video_path
                ]
                import json
                probe = subprocess.run(
                    probe_cmd, capture_output=True,
                    encoding='utf-8', errors='replace'
                )
                info = json.loads(probe.stdout)
                src_w = src_h = None
                for s in info.get("streams", []):
                    if s.get("codec_type") == "video":
                        src_w, src_h = int(s["width"]), int(s["height"])
                        break

                if src_w and src_h:
                    tw, th = vertical_width, vertical_height
                    filter_complex = (
                        f"[0:v]split=2[bg_in][fg_in];"
                        f"[bg_in]scale={tw}:{th}:force_original_aspect_ratio=increase,"
                        f"crop={tw}:{th},gblur=sigma=40,eq=brightness=-0.08[bg];"
                        f"[fg_in]scale={tw}:{th}:force_original_aspect_ratio=decrease,"
                        f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black@0[fg];"
                        f"[bg][fg]overlay=(W-w)/2:(H-h)/2:format=auto[vout]"
                    )
                    cmd.extend(["-filter_complex", filter_complex])
                    cmd.extend(["-map", "[vout]", "-map", "0:a?"])
            except Exception as e:
                logger.warning(f"Could not set up vertical filter: {e}")

        cmd.extend([
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            "-movflags", "+faststart",
            output_path
        ])

        try:
            result = subprocess.run(
                cmd, capture_output=True, encoding='utf-8',
                errors='replace', timeout=120
            )

            if result.returncode != 0:
                error_msg = result.stderr[-300:] if result.stderr else "Unknown"
                results.append({
                    "clip_number": i,
                    "filename": output_name,
                    "success": False,
                    "error": error_msg,
                })
            else:
                out_file = Path(output_path)
                size_mb = out_file.stat().st_size / (1024*1024) if out_file.exists() else 0
                results.append({
                    "clip_number": i,
                    "filename": output_name,
                    "output_path": output_path,
                    "start": clip.start,
                    "end": clip.end,
                    "duration": clip.duration,
                    "label": clip.label,
                    "size_mb": round(size_mb, 2),
                    "success": True,
                })
                logger.info(f"  -> Saved: {output_name} ({size_mb:.1f} MB)")

        except subprocess.TimeoutExpired:
            results.append({
                "clip_number": i,
                "filename": output_name,
                "success": False,
                "error": "FFmpeg timed out",
            })

    success = sum(1 for r in results if r.get("success"))
    logger.info(f"Manual split complete: {success}/{len(results)} succeeded")
    return results
