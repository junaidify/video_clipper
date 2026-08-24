"""
Video Clipper Module
Executes FFmpeg cutting, 9:16 vertical scale-to-fit transformation with blurred
background fill, audio/video fade transitions, anti-copyright modulation, and batch rendering.
"""
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Callable

from video_clipper.config import ClipperConfig
from video_clipper.clipping.analyzer import ClipCandidate

logger = logging.getLogger(__name__)


@dataclass
class ClipResult:
    """Result metadata for a generated clip."""
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
    file_size_mb: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class VideoClipper:
    """Manages video cutting, vertical scaling, and clip generation via FFmpeg."""

    def __init__(self, config: Optional[ClipperConfig] = None):
        """
        Args:
            config: ClipperConfig instance (defaults if None).
        """
        self.config = config or ClipperConfig()
        self._verify_ffmpeg()

    @staticmethod
    def _verify_ffmpeg():
        """Verify that ffmpeg and ffprobe are available on the system PATH."""
        for tool in ["ffmpeg", "ffprobe"]:
            try:
                result = subprocess.run(
                    [tool, "-version"],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                if result.returncode != 0:
                    raise RuntimeError(f"'{tool}' returned non-zero exit code.")
            except FileNotFoundError:
                raise RuntimeError(
                    f"'{tool}' is not installed or not found on PATH. "
                    "Please install FFmpeg: https://ffmpeg.org/download.html"
                )

    def get_video_info(self, video_path: str) -> dict:
        """
        Extract video dimensions, duration, fps, and audio streams via ffprobe.

        Returns:
            Dict with width, height, duration, fps, has_audio keys.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        data = json.loads(result.stdout)

        width = 1920
        height = 1080
        fps = 30.0
        has_audio = False

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                width = int(stream.get("width", 1920))
                height = int(stream.get("height", 1080))
                # Parse r_frame_rate (e.g. "30/1" or "29.97/1")
                r_fps = stream.get("r_frame_rate", "30/1")
                try:
                    num, den = r_fps.split("/")
                    fps = float(num) / float(den) if float(den) > 0 else 30.0
                except Exception:
                    fps = 30.0
            elif stream.get("codec_type") == "audio":
                has_audio = True

        duration = float(data.get("format", {}).get("duration", 0.0))

        return {
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps,
            "has_audio": has_audio,
        }

    def _adjust_boundaries(
        self,
        candidate: ClipCandidate,
        video_duration: float,
    ) -> Tuple[float, float]:
        """
        Apply pre/post padding and enforce min/max clip duration constraints.
        """
        cfg = self.config

        # Apply padding
        start = max(0.0, candidate.start - cfg.pre_hook_padding)
        end = min(video_duration, candidate.end + cfg.post_hook_padding)
        duration = end - start

        # If too short, expand forward then backward
        if duration < cfg.min_clip_duration:
            shortfall = cfg.min_clip_duration - duration
            end = min(video_duration, end + shortfall)
            duration = end - start
            if duration < cfg.min_clip_duration:
                start = max(0.0, start - (cfg.min_clip_duration - duration))

        # If too long, clamp end
        if (end - start) > cfg.max_clip_duration:
            end = start + cfg.max_clip_duration

        return round(start, 2), round(end, 2)

    def _build_vertical_filter_complex(
        self,
        src_w: int,
        src_h: int,
        duration: float,
        anti_copyright: bool = False,
        anti_copyright_config: Optional[dict] = None,
    ) -> str:
        """
        Build the FFmpeg filter_complex string for:
        1. Background: scale to fit 9:16 vertical canvas (1080x1920), crop, and heavy gaussian blur.
        2. Foreground: scale original content to fit canvas width while preserving aspect ratio.
        3. Overlay: center foreground on top of blurred background.
        4. Optional: Anti-copyright subtle pixel modulation (zoom, hflip, color grade, grain).
        5. Fade in/out transitions.
        """
        tw = self.config.vertical_width
        th = self.config.vertical_height
        fade_d = self.config.fade_duration

        # Background branch: scale to fill canvas, center-crop, blur, and darken
        bg_filter = (
            f"[bg_in]scale={tw}:{th}:force_original_aspect_ratio=increase,"
            f"crop={tw}:{th},gblur=sigma=60,"
            f"eq=brightness=-0.12:saturation=1.3,vignette=PI/3.5[bg]"
        )

        # Foreground branch: scale to fit width/height preserving aspect ratio
        fg_filter = f"[fg_in]scale={tw}:{th}:force_original_aspect_ratio=decrease[fg]"

        # Overlay composite
        overlay_filter = f"[bg][fg]overlay=(W-w)/2:(H-h)/2:format=auto[comp]"

        current_output = "[comp]"
        extra_filters = []

        # Anti-copyright modulation if requested
        if anti_copyright:
            opts = anti_copyright_config or {}
            mod_filters = []

            # 1. Subtle zoom (3-5%)
            zoom = opts.get("zoom_percent", 3.0)
            if zoom > 0:
                scale_f = 1.0 + (zoom / 100.0)
                mod_filters.append(f"scale=iw*{scale_f}:ih*{scale_f},crop=iw/{scale_f}:ih/{scale_f}")

            # 2. Horizontal mirror flip
            if opts.get("mirror_flip", False):
                mod_filters.append("hflip")

            # 3. Color grade
            grade = opts.get("color_grade", "warm")
            if grade == "warm":
                mod_filters.append("eq=brightness=0.02:contrast=1.05:saturation=1.12:gamma_r=1.05:gamma_b=0.95")
            elif grade == "cool":
                mod_filters.append("eq=brightness=0.01:contrast=1.08:saturation=1.05:gamma_r=0.95:gamma_b=1.08")
            elif grade == "cinematic":
                mod_filters.append("eq=brightness=-0.02:contrast=1.15:saturation=0.90:gamma_r=1.02:gamma_b=1.02")
            elif grade == "vibrant":
                mod_filters.append("eq=brightness=0.03:contrast=1.10:saturation=1.25")

            # 4. Film grain noise
            if opts.get("grain_overlay", True):
                intensity = int(opts.get("grain_intensity", 0.04) * 255)
                mod_filters.append(f"noise=alls={intensity}:allf=t")

            if mod_filters:
                extra_filters.append(f"{current_output}{','.join(mod_filters)}[mod]")
                current_output = "[mod]"

        # Fade in/out
        if fade_d > 0 and duration > (fade_d * 2):
            fade_out_start = max(0.0, duration - fade_d)
            extra_filters.append(
                f"{current_output}fade=t=in:st=0:d={fade_d},"
                f"fade=t=out:st={fade_out_start:.2f}:d={fade_d}[vout]"
            )
        else:
            if current_output != "[vout]":
                extra_filters.append(f"{current_output}copy[vout]")

        chain = f"[0:v]split=2[bg_in][fg_in];{bg_filter};{fg_filter};{overlay_filter}"
        if extra_filters:
            chain += ";" + ";".join(extra_filters)

        return chain

    def _build_fade_filter(self, duration: float) -> str:
        """Build simple video fade-in/fade-out filter for non-vertical cutting."""
        fade_d = self.config.fade_duration
        if fade_d > 0 and duration > (fade_d * 2):
            fade_out_start = max(0.0, duration - fade_d)
            return f"fade=t=in:st=0:d={fade_d},fade=t=out:st={fade_out_start:.2f}:d={fade_d}"
        return ""

    def cut_clip(
        self,
        video_path: str,
        start: float,
        end: float,
        output_path: str,
        src_info: Optional[dict] = None,
        anti_copyright: bool = False,
        anti_copyright_config: Optional[dict] = None,
    ) -> bool:
        """
        Cut a single clip from a video file with frame accuracy and optional vertical scaling.

        Args:
            video_path: Source video path.
            start: Start timestamp in seconds.
            end: End timestamp in seconds.
            output_path: Destination path for the clip.
            src_info: Optional pre-probed video info.
            anti_copyright: Enable anti-copyright modulation filters.
            anti_copyright_config: Configuration dictionary for anti-copyright filters.

        Returns:
            True if output was created successfully, False otherwise.
        """
        if not src_info:
            src_info = self.get_video_info(video_path)

        duration = end - start
        if duration <= 0:
            logger.error(f"Invalid clip duration: {duration}s (start={start}, end={end})")
            return False

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cfg = self.config

        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",                   # overwrite output
            "-ss", str(start),      # fast input seeking
            "-i", video_path,
            "-t", str(duration),    # duration to extract
            "-avoid_negative_ts", "1",
        ]

        # Video filters
        if cfg.crop_vertical:
            filter_complex = self._build_vertical_filter_complex(
                src_info["width"],
                src_info["height"],
                duration,
                anti_copyright=anti_copyright,
                anti_copyright_config=anti_copyright_config,
            )
            cmd.extend(["-filter_complex", filter_complex])
            cmd.extend(["-map", "[vout]"])
            if src_info["has_audio"]:
                # Audio filter for anti-copyright pitch/speed shift or fade
                audio_filters = []
                fade_d = cfg.fade_duration
                if fade_d > 0 and duration > (fade_d * 2):
                    fade_out_start = max(0.0, duration - fade_d)
                    audio_filters.append(f"afade=t=in:st=0:d={fade_d},afade=t=out:st={fade_out_start:.2f}:d={fade_d}")

                if anti_copyright and anti_copyright_config:
                    speed = anti_copyright_config.get("speed_shift", 1.0)
                    if speed != 1.0 and 0.5 <= speed <= 2.0:
                        audio_filters.append(f"atempo={speed}")

                if audio_filters:
                    cmd.extend(["-af", ",".join(audio_filters)])
                cmd.extend(["-map", "0:a"])
        else:
            vf = self._build_fade_filter(duration)
            if vf:
                cmd.extend(["-vf", vf])
            if src_info["has_audio"]:
                cmd.extend(["-map", "0:a?"])

        # Encoding parameters
        cmd.extend([
            "-c:v", cfg.video_codec,
            "-preset", cfg.video_preset,
            "-crf", str(cfg.video_quality),
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",   # Ensure wide player compatibility (iOS, Web, Android)
            "-threads", "0",          # Use all CPU cores
        ])

        if src_info["has_audio"]:
            cmd.extend([
                "-c:a", cfg.audio_codec,
                "-b:a", cfg.audio_bitrate,
                "-ar", "44100",
            ])

        cmd.extend([
            "-movflags", "+faststart",  # Enable immediate playback streaming
            output_path
        ])

        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300  # 5 min timeout per clip
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg cutting failed: {result.stderr[-400:]}")
            return False

        return os.path.isfile(output_path) and os.path.getsize(output_path) > 1000

    def process_clips(
        self,
        video_path: str,
        candidates: List[ClipCandidate],
        output_dir: str,
        anti_copyright: bool = False,
        anti_copyright_config: Optional[dict] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[ClipResult]:
        """
        Process a list of ClipCandidate objects into final video clips on disk.

        Args:
            video_path: Source video path.
            candidates: List of ClipCandidate objects from ContentAnalyzer.
            output_dir: Directory to store generated clips.
            anti_copyright: Apply anti-copyright modulation transforms.
            anti_copyright_config: Modulation settings.
            progress_callback: Optional fn(current_idx, total_clips, status_msg).

        Returns:
            List of ClipResult objects.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        src_info = self.get_video_info(video_path)
        video_dur = src_info["duration"]
        video_stem = Path(video_path).stem

        results = []
        total = len(candidates)

        logger.info(f"Processing {total} clips into {output_dir}...")

        for i, candidate in enumerate(candidates, 1):
            start, end = self._adjust_boundaries(candidate, video_dur)
            duration = end - start

            out_filename = f"clip_{i:02d}_{start:.0f}s-{end:.0f}s.{self.config.output_format}"
            out_filepath = str(out_dir / out_filename)

            msg = f"Rendering clip {i}/{total}: {start:.1f}s -> {end:.1f}s ({duration:.1f}s)"
            logger.info(msg)
            if progress_callback:
                progress_callback(i, total, msg)

            success = self.cut_clip(
                video_path,
                start=start,
                end=end,
                output_path=out_filepath,
                src_info=src_info,
                anti_copyright=anti_copyright,
                anti_copyright_config=anti_copyright_config,
            )

            file_size_mb = 0.0
            error = None
            if success and os.path.isfile(out_filepath):
                file_size_mb = round(os.path.getsize(out_filepath) / (1024 * 1024), 2)
            else:
                error = "FFmpeg cutting failed or produced empty file"

            results.append(ClipResult(
                clip_number=i,
                output_path=out_filepath,
                start=start,
                end=end,
                duration=duration,
                score=candidate.score,
                reason=candidate.reason,
                hook_text=candidate.hook_text,
                success=success,
                error=error,
                file_size_mb=file_size_mb,
            ))

        successful = sum(1 for r in results if r.success)
        logger.info(f"Clipping finished: {successful}/{total} clips generated successfully.")
        return results
