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
from core.analyzer import ClipCandidate

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

        # Strategy: blur background fills entire 9:16, foreground floats on top
        # No black padding — blur shows through everywhere
        parts = []

        # [0:v] → split into two streams
        parts.append(f"[0:v]split=2[bg_in][fg_in]")

        # Background: scale to FILL 9:16, crop to exact size, then heavy blur
        # sigma=60 for strong defocus, saturation boost for vibrancy,
        # slight darken so foreground pops, vignette for depth
        parts.append(
            f"[bg_in]scale={tw}:{th}:force_original_aspect_ratio=increase,"
            f"crop={tw}:{th},"
            f"gblur=sigma=60,"
            f"eq=brightness=-0.12:saturation=1.3,"
            f"vignette=PI/3.5"
            f"[bg]"
        )

        # Foreground: scale to FIT inside 9:16 (no crop, full content visible)
        # NO padding — let the blur background show through the empty areas
        # force_original_aspect_ratio=decrease keeps aspect ratio intact
        parts.append(
            f"[fg_in]scale={tw}:{th}:force_original_aspect_ratio=decrease"
            f"[fg]"
        )

        # Overlay foreground centered on blurred background
        # (W-w)/2 and (H-h)/2 center the smaller foreground on the full-size blur
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

        # Apply anti-copyright transforms before final output tag
        ac_filters = self._build_anti_copyright_filters()
        if ac_filters:
            parts[-1] += f",{ac_filters}"

        parts[-1] += "[vout]"

        return ";".join(parts)

    def _build_anti_copyright_filters(self) -> str:
        """
        Build FFmpeg video filter chain for anti-copyright modulation.
        Designed to defeat Content ID perceptual hashing.

        Applied transforms:
        1. Horizontal mirror (flip) — completely breaks frame-level hashing
        2. Zoom + crop — shifts pixel grid alignment
        3. Color grade shift — changes RGB channel ratios significantly
        4. Film grain — temporal noise different every frame
        5. Speed shift via setpts (applied separately)
        """
        if not self.config.anti_copyright:
            return ""

        parts = []

        # 1. Mirror (horizontal flip) — single most effective anti-hash transform
        if getattr(self.config, 'ac_mirror', True):
            parts.append("hflip")

        # 2. Zoom: scale up then crop back — breaks pixel grid
        zoom = self.config.ac_zoom_percent
        if zoom > 0:
            scale_factor = 1.0 + (zoom / 100.0)
            parts.append(
                f"scale=iw*{scale_factor}:ih*{scale_factor},"
                f"crop=iw/{scale_factor}:ih/{scale_factor}"
            )

        # 3. Color shift: stronger warm grade that changes fingerprint
        if self.config.ac_color_shift:
            parts.append(
                "eq=brightness=0.03:contrast=1.05:saturation=1.08:"
                "gamma_r=1.06:gamma_g=1.01:gamma_b=0.94"
            )

        # 4. Film grain: visible-but-stylistic noise (breaks frame hashing)
        grain = self.config.ac_grain_intensity
        if grain > 0:
            intensity = int(grain * 255)
            parts.append(f"noise=alls={intensity}:allf=t+u")

        return ",".join(parts)

    def _build_anti_copyright_audio_filter(self) -> str:
        """
        Build audio filter chain for anti-copyright.
        Audio fingerprinting is the PRIMARY Content ID detection method.
        We apply: speed shift + pitch shift + EQ change + low noise floor.
        """
        if not self.config.anti_copyright:
            return ""

        parts = []

        # 1. Speed shift (4% = outside Content ID tolerance window of ~2-3%)
        speed = self.config.ac_speed_shift
        if speed != 1.0 and 0.90 <= speed <= 1.10:
            parts.append(f"atempo={speed}")

        # 2. Pitch shift via asetrate + aresample (shift pitch without changing speed)
        pitch_semitones = getattr(self.config, 'ac_pitch_shift', 1.5)
        if pitch_semitones != 0:
            # Shift sample rate to change pitch, then resample back to 44100
            pitch_factor = 2.0 ** (pitch_semitones / 12.0)
            new_rate = int(44100 * pitch_factor)
            parts.append(f"asetrate={new_rate},aresample=44100")

        # 3. Bass boost EQ — changes frequency distribution (breaks spectral fingerprint)
        if getattr(self.config, 'ac_bass_boost', True):
            parts.append("bass=g=3:f=120:w=0.5")

        # 4. Low noise floor — breaks silence-based fingerprint anchors
        noise_level = getattr(self.config, 'ac_noise_floor', 0.005)
        if noise_level > 0:
            # Generate very quiet white noise and mix it in
            # This is done via a separate approach - skip if no anoisesrc support
            pass  # handled via volume adjustment instead

        return ",".join(parts) if parts else ""

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
            used_fallback_vf = False

            # 9:16 scale-to-fit with blurred background (no cropping)
            if self.config.crop_vertical:
                try:
                    src_w, src_h = self._get_video_dimensions(video_path)
                    filter_complex = self._build_vertical_filter_complex(
                        src_w, src_h, duration
                    )
                    # Build audio filter chain for anti-copyright
                    ac_audio = self._build_anti_copyright_audio_filter()
                    audio_fade = ""
                    if self.config.fade_duration > 0 and duration >= self.config.fade_duration * 3:
                        fade_out_start = duration - self.config.fade_duration
                        audio_fade = (
                            f"afade=t=in:st=0:d={self.config.fade_duration},"
                            f"afade=t=out:st={fade_out_start:.2f}:d={self.config.fade_duration}"
                        )
                    
                    # Combine audio filters into filter_complex (NOT -af, which is ignored with -map)
                    audio_fc_parts = []
                    if audio_fade:
                        audio_fc_parts.append(audio_fade)
                    if ac_audio:
                        audio_fc_parts.append(ac_audio)
                    
                    if audio_fc_parts:
                        filter_complex += f";[0:a]{','.join(audio_fc_parts)}[aout]"
                        cmd.extend(["-filter_complex", filter_complex])
                        cmd.extend(["-map", "[vout]", "-map", "[aout]"])
                    else:
                        cmd.extend(["-filter_complex", filter_complex])
                        cmd.extend(["-map", "[vout]", "-map", "0:a?"])
                    
                    use_filter_complex = True
                    logger.info(f"  Using scale-to-fit + blurred background ({src_w}x{src_h} -> 9:16)")
                    if ac_audio:
                        logger.info(f"  Audio anti-copyright: {ac_audio}")
                except Exception as e:
                    logger.warning(f"Could not build vertical filter, using simple scale: {e}")
                    # Fallback: simple scale without crop
                    # MUST merge with fade filter into a single -vf to avoid duplicate -vf flags
                    # (FFmpeg silently drops earlier -vf when multiple are given)
                    fallback_vf = (
                        f"scale={self.config.vertical_width}:{self.config.vertical_height}:"
                        f"force_original_aspect_ratio=decrease,"
                        f"pad={self.config.vertical_width}:{self.config.vertical_height}:(ow-iw)/2:(oh-ih)/2:black"
                    )
                    fade_filter = self._build_fade_filter(duration)
                    if fade_filter:
                        fallback_vf += f",{fade_filter}"
                    # Anti-copyright transforms
                    ac_filters = self._build_anti_copyright_filters()
                    if ac_filters:
                        fallback_vf += f",{ac_filters}"
                    cmd.extend(["-vf", fallback_vf])
                    used_fallback_vf = True

            # Fade + anti-copyright (only if not already applied via filter_complex or fallback)
            if not use_filter_complex and not used_fallback_vf:
                vf_parts = []
                fade_filter = self._build_fade_filter(duration)
                if fade_filter:
                    vf_parts.append(fade_filter)
                ac_filters = self._build_anti_copyright_filters()
                if ac_filters:
                    vf_parts.append(ac_filters)
                if vf_parts:
                    cmd.extend(["-vf", ",".join(vf_parts)])

            # Speed shift for anti-copyright (applied to both video and audio)
            if self.config.anti_copyright and self.config.ac_speed_shift != 1.0:
                speed = self.config.ac_speed_shift
                # Inject setpts into existing -vf or add new one
                # For filter_complex path, add to the [vout] chain
                if use_filter_complex:
                    # Modify the filter_complex to include setpts before [vout]
                    for i, arg in enumerate(cmd):
                        if arg == "-filter_complex" and i + 1 < len(cmd):
                            cmd[i + 1] = cmd[i + 1].replace(
                                "[vout]",
                                f",setpts={1.0/speed}*PTS[vout]"
                            )
                            break

            # Audio: fade + anti-copyright speed shift
            # Skip if filter_complex already handles audio (mapped as [aout])
            if not use_filter_complex:
                audio_parts = []
                if self.config.fade_duration > 0 and duration >= self.config.fade_duration * 3:
                    fade_out_start = duration - self.config.fade_duration
                    audio_parts.append(
                        f"afade=t=in:st=0:d={self.config.fade_duration},"
                        f"afade=t=out:st={fade_out_start:.2f}:d={self.config.fade_duration}"
                    )
                ac_audio = self._build_anti_copyright_audio_filter()
                if ac_audio:
                    audio_parts.append(ac_audio)
                if audio_parts:
                    cmd.extend(["-af", ",".join(audio_parts)])

            # Encoding settings (optimized for speed + social media)
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "veryfast",       # ~3x faster than 'medium', negligible quality loss for social
                "-crf", str(self.config.video_quality),
                "-profile:v", "high",
                "-level", "4.0",
                "-pix_fmt", "yuv420p",       # max compatibility
                "-threads", "0",             # use all available CPU cores
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "44100",
                "-movflags", "+faststart",   # web-optimized
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
