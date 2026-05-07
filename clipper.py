"""
Video Clipper Module
Actually splits the video into short clips using FFmpeg.

Features:
- Smart boundary adjustment
- 9:16 vertical fit with blurred background fill (YouTube Shorts style)
  — no content is cropped or lost, video shrinks to fit
- Fade in/out transitions
- Quality-optimized encoding for social platforms
- Duration enforcement (min/max bounds)
"""
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import ClipperConfig
from analyzer import ClipCandidate

logger = logging.getLogger(__name__)


@dataclass
class ClipResult:
    """Result of a single clip extraction."""
    clip_number: int
    output_path: str
    start: float
    end: float
    duration: float
    score: float
    reason: str
    hook_text: str
    success: bool
    error: Optional[str] = None


class VideoClipper:
    """Cuts video into short clips using FFmpeg."""

    def __init__(self, config: Optional[ClipperConfig] = None):
        self.config = config or ClipperConfig()
        self._verify_ffmpeg()

    def _verify_ffmpeg(self):
        """Check that ffmpeg is available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, check=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError(
                "FFmpeg is required but not found. Install it:\n"
                "  Windows: winget install ffmpeg\n"
                "  macOS: brew install ffmpeg\n"
                "  Linux: sudo apt install ffmpeg"
            )

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            video_path
        ]
        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8', errors='replace', check=True
        )
        return json.loads(result.stdout)

    def _get_video_dimensions(self, video_path: str) -> tuple:
        """Get video width and height."""
        info = self.get_video_info(video_path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                return int(stream["width"]), int(stream["height"])
        raise ValueError("No video stream found")

    def _build_vertical_filter_complex(self, src_width: int, src_height: int,
                                       duration: float) -> str:
        """
        Build FFmpeg filter_complex for 9:16 output WITHOUT cropping.

        Strategy (YouTube Shorts style):
        1. Background layer: scale source to FILL 9:16, then heavy gaussian blur
        2. Foreground layer: scale source to FIT inside 9:16 (no crop, full content visible)
        3. Overlay foreground centered on blurred background
        4. Apply fade transitions

        Result: all content is visible, empty space filled with blurred video.
        """
        tw = self.config.vertical_width   # 1080
        th = self.config.vertical_height  # 1920

        # Background: scale to fill 9:16 (will crop to fill, but it's just the blur layer)
        # Foreground: scale to fit inside 9:16 (no content lost)
        parts = []

        # [0:v] → split into two streams
        parts.append(f"[0:v]split=2[bg_in][fg_in]")

        # Background: scale to fill + crop to exact 9:16 + blur heavily
        parts.append(
            f"[bg_in]scale={tw}:{th}:force_original_aspect_ratio=increase,"
            f"crop={tw}:{th},"
            f"gblur=sigma=40,"
            f"eq=brightness=-0.08"
            f"[bg]"
        )

        # Foreground: scale to fit inside 9:16 (maintains aspect ratio, no crop)
        # force_original_aspect_ratio=decrease ensures it fits within the box
        # pad ensures even dimensions
        parts.append(
            f"[fg_in]scale={tw}:{th}:force_original_aspect_ratio=decrease,"
            f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black@0"
            f"[fg]"
        )

        # Overlay foreground on blurred background, centered
        parts.append(
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2:format=auto"
        )

        # Add fade if configured
        fade_dur = self.config.fade_duration
        if fade_dur > 0 and duration >= fade_dur * 3:
            fade_out_start = duration - fade_dur
            parts[-1] += (
                f",fade=t=in:st=0:d={fade_dur},"
                f"fade=t=out:st={fade_out_start:.2f}:d={fade_dur}"
            )

        parts[-1] += "[vout]"

        return ";".join(parts)

    def _build_fade_filter(self, duration: float) -> str:
        """Build fade in/out filter."""
        fade_dur = self.config.fade_duration
        if fade_dur <= 0 or duration < fade_dur * 3:
            return ""

        fade_out_start = duration - fade_dur
        return (
            f"fade=t=in:st=0:d={fade_dur},"
            f"fade=t=out:st={fade_out_start:.2f}:d={fade_dur}"
        )

    def _adjust_boundaries(self, candidate: ClipCandidate,
                           video_duration: float) -> tuple:
        """
        Adjust clip boundaries to enforce duration limits and add padding.
        Returns (adjusted_start, adjusted_end).
        """
        cfg = self.config

        # Add padding
        start = max(0, candidate.start - cfg.pre_hook_padding)
        end = min(video_duration, candidate.end + cfg.post_hook_padding)

        duration = end - start

        # Enforce minimum duration
        if duration < cfg.min_clip_duration:
            deficit = cfg.min_clip_duration - duration
            start = max(0, start - deficit / 2)
            end = min(video_duration, end + deficit / 2)
            duration = end - start

        # Enforce maximum duration
        if duration > cfg.max_clip_duration:
            # Keep centered on the original hook
            center = (candidate.start + candidate.end) / 2
            half = cfg.max_clip_duration / 2
            start = max(0, center - half)
            end = min(video_duration, start + cfg.max_clip_duration)
            start = max(0, end - cfg.max_clip_duration)

        return round(start, 2), round(end, 2)

    def extract_clip(self, video_path: str, candidate: ClipCandidate,
                     clip_number: int, output_dir: str,
                     video_duration: float) -> ClipResult:
        """
        Extract a single clip from the source video.

        This is where the actual video splitting happens.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Adjust boundaries
        start, end = self._adjust_boundaries(candidate, video_duration)
        duration = end - start

        # Build output filename
        score_tag = f"{candidate.score:.2f}".replace(".", "")
        output_name = f"clip_{clip_number:02d}_score{score_tag}_{start:.0f}s-{end:.0f}s.{self.config.output_format}"
        output_path = str(output_dir / output_name)

        logger.info(
            f"Extracting clip {clip_number}: "
            f"{start:.1f}s - {end:.1f}s ({duration:.1f}s) "
            f"[score={candidate.score:.3f}]"
        )

        try:
            # Build FFmpeg command
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),          # seek to start (before -i for fast seek)
                "-i", video_path,
                "-t", str(duration),         # duration
                "-avoid_negative_ts", "1",
            ]

            use_filter_complex = False

            # 9:16 scale-to-fit with blurred background (no cropping)
            if self.config.crop_vertical:
                try:
                    src_w, src_h = self._get_video_dimensions(video_path)
                    filter_complex = self._build_vertical_filter_complex(
                        src_w, src_h, duration
                    )
                    cmd.extend(["-filter_complex", filter_complex])
                    cmd.extend(["-map", "[vout]", "-map", "0:a?"])
                    use_filter_complex = True
                    logger.info(f"  Using scale-to-fit + blurred background ({src_w}x{src_h} -> 9:16)")
                except Exception as e:
                    logger.warning(f"Could not build vertical filter, using simple scale: {e}")
                    # Fallback: simple scale without crop
                    cmd.extend(["-vf",
                        f"scale={self.config.vertical_width}:{self.config.vertical_height}:"
                        f"force_original_aspect_ratio=decrease,"
                        f"pad={self.config.vertical_width}:{self.config.vertical_height}:(ow-iw)/2:(oh-ih)/2:black"
                    ])

            # Fade (only if not already applied via filter_complex)
            if not use_filter_complex:
                fade_filter = self._build_fade_filter(duration)
                if fade_filter:
                    cmd.extend(["-vf", fade_filter])

            # Audio fade
            if self.config.fade_duration > 0 and duration >= self.config.fade_duration * 3:
                fade_out_start = duration - self.config.fade_duration
                audio_fade = (
                    f"afade=t=in:st=0:d={self.config.fade_duration},"
                    f"afade=t=out:st={fade_out_start:.2f}:d={self.config.fade_duration}"
                )
                cmd.extend(["-af", audio_fade])

            # Encoding settings (optimized for social media)
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", str(self.config.video_quality),
                "-profile:v", "high",
                "-level", "4.0",
                "-pix_fmt", "yuv420p",       # max compatibility
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "44100",
                "-movflags", "+faststart",    # web-optimized
                output_path
            ])

            # Execute FFmpeg (utf-8 encoding to handle non-ASCII filenames)
            result = subprocess.run(
                cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=120
            )

            if result.returncode != 0:
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                logger.error(f"FFmpeg failed for clip {clip_number}: {error_msg}")
                return ClipResult(
                    clip_number=clip_number,
                    output_path=output_path,
                    start=start, end=end,
                    duration=duration,
                    score=candidate.score,
                    reason=candidate.reason,
                    hook_text=candidate.hook_text,
                    success=False,
                    error=error_msg,
                )

            # Verify output file exists and has size
            out_file = Path(output_path)
            if not out_file.exists() or out_file.stat().st_size < 1000:
                return ClipResult(
                    clip_number=clip_number,
                    output_path=output_path,
                    start=start, end=end,
                    duration=duration,
                    score=candidate.score,
                    reason=candidate.reason,
                    hook_text=candidate.hook_text,
                    success=False,
                    error="Output file missing or too small",
                )

            file_size_mb = out_file.stat().st_size / (1024 * 1024)
            logger.info(f"  -> Saved: {output_path} ({file_size_mb:.1f} MB)")

            return ClipResult(
                clip_number=clip_number,
                output_path=output_path,
                start=start, end=end,
                duration=duration,
                score=candidate.score,
                reason=candidate.reason,
                hook_text=candidate.hook_text,
                success=True,
            )

        except subprocess.TimeoutExpired:
            return ClipResult(
                clip_number=clip_number,
                output_path=output_path,
                start=start, end=end,
                duration=duration,
                score=candidate.score,
                reason=candidate.reason,
                hook_text=candidate.hook_text,
                success=False,
                error="FFmpeg timed out (>120s)",
            )
        except Exception as e:
            return ClipResult(
                clip_number=clip_number,
                output_path=output_path,
                start=start, end=end,
                duration=duration,
                score=candidate.score,
                reason=candidate.reason,
                hook_text=candidate.hook_text,
                success=False,
                error=str(e),
            )

    def extract_all_clips(self, video_path: str, candidates: list,
                          output_dir: str, video_duration: float) -> list:
        """
        Extract all clip candidates from the video.
        Returns list of ClipResult objects.
        """
        if not candidates:
            logger.warning("No clip candidates to extract.")
            return []

        logger.info(f"Extracting {len(candidates)} clips from: {video_path}")
        logger.info(f"Output directory: {output_dir}")

        results = []
        for i, candidate in enumerate(candidates, 1):
            result = self.extract_clip(
                video_path=video_path,
                candidate=candidate,
                clip_number=i,
                output_dir=output_dir,
                video_duration=video_duration,
            )
            results.append(result)

        # Summary
        success_count = sum(1 for r in results if r.success)
        fail_count = sum(1 for r in results if not r.success)
        logger.info(f"Extraction complete: {success_count} succeeded, {fail_count} failed")

        return results

    def save_results(self, results: list, path: str):
        """Save extraction results to JSON."""
        data = []
        for r in results:
            data.append({
                "clip_number": r.clip_number,
                "output_path": r.output_path,
                "start": r.start,
                "end": r.end,
                "duration": r.duration,
                "score": r.score,
                "reason": r.reason,
                "hook_text": r.hook_text[:200],
                "success": r.success,
                "error": r.error,
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {path}")
