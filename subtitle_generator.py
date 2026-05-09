"""
Subtitle Generator Module
Generates SRT subtitles from video using Whisper (local) with Gemini fallback.
Burns subtitles into video using FFmpeg.
"""

import json
import logging
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_font_path() -> str:
    """Get a valid bold font path for the current OS."""
    if platform.system() == "Windows":
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        ]
    for p in candidates:
        if os.path.isfile(p):
            return p.replace("\\", "/")
    return ""  # let FFmpeg try its built-in default


def _get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return 0.0


def _get_video_dimensions(video_path: str) -> tuple:
    """Get video width and height."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception:
        return 1080, 1920


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe_with_whisper(video_path: str, model_size: str = "base") -> list:
    """
    Transcribe video using local Whisper and return word-level timestamps.

    Returns:
        List of dicts: [{"word": "hello", "start": 0.0, "end": 0.5}, ...]
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("Whisper not installed. pip install openai-whisper")

    # Extract audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path
        ]
        subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', check=True)

        model = whisper.load_model(model_size, device="cpu")
        result = model.transcribe(audio_path, word_timestamps=True, verbose=False)

        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                })

        logger.info(f"Whisper transcribed {len(words)} words from {video_path}")
        return words
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)


def transcribe_with_gemini(video_path: str) -> list:
    """
    Fallback: transcribe using Gemini API.
    Extracts audio, sends to Gemini for transcription with timestamps.

    Returns:
        List of dicts: [{"word": "hello", "start": 0.0, "end": 0.5}, ...]
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Install google-generativeai: pip install google-generativeai")

    # Extract audio as mp3 (smaller for upload)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        audio_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1",
            "-b:a", "64k", audio_path
        ]
        subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', check=True)

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Upload audio file
        audio_file = genai.upload_file(audio_path, mime_type="audio/mpeg")

        prompt = """Transcribe this audio with word-level timestamps.
Return ONLY a JSON array, no markdown, no explanation.
Each element: {"word": "the_word", "start": seconds_float, "end": seconds_float}
Group into natural phrases of 4-8 words for subtitle display.
Example: [{"word": "Hello world", "start": 0.0, "end": 0.8}, ...]"""

        response = model.generate_content([prompt, audio_file])
        text = response.text.strip()

        # Clean markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        words = json.loads(text)
        logger.info(f"Gemini transcribed {len(words)} segments from {video_path}")
        return words
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)


def words_to_srt(words: list, max_words_per_line: int = 6, max_duration: float = 4.0) -> str:
    """
    Convert word-level timestamps to SRT subtitle format.
    Groups words into subtitle lines.
    """
    if not words:
        return ""

    subtitles = []
    current_words = []
    current_start = None

    for w in words:
        if current_start is None:
            current_start = w["start"]
        current_words.append(w["word"])

        # Check if we should break the line
        elapsed = w["end"] - current_start
        if len(current_words) >= max_words_per_line or elapsed >= max_duration:
            subtitles.append({
                "index": len(subtitles) + 1,
                "start": current_start,
                "end": w["end"],
                "text": " ".join(current_words),
            })
            current_words = []
            current_start = None

    # Remaining words
    if current_words:
        subtitles.append({
            "index": len(subtitles) + 1,
            "start": current_start,
            "end": words[-1]["end"],
            "text": " ".join(current_words),
        })

    # Build SRT string
    srt_lines = []
    for sub in subtitles:
        srt_lines.append(str(sub["index"]))
        srt_lines.append(f"{_format_srt_time(sub['start'])} --> {_format_srt_time(sub['end'])}")
        srt_lines.append(sub["text"])
        srt_lines.append("")

    return "\n".join(srt_lines)


def generate_subtitles(video_path: str, model_size: str = "base") -> dict:
    """
    Generate subtitles for a video. Tries Whisper locally, falls back to Gemini.

    Returns:
        {"words": [...], "srt": "...", "method": "whisper"|"gemini"}
    """
    video_path = str(Path(video_path).resolve())

    # Try Whisper first
    try:
        logger.info(f"Attempting Whisper transcription (model={model_size})")
        words = transcribe_with_whisper(video_path, model_size)
        if words:
            srt = words_to_srt(words)
            return {"words": words, "srt": srt, "method": "whisper"}
    except Exception as e:
        logger.warning(f"Whisper failed: {e}")

    # Fallback to Gemini
    try:
        logger.info("Falling back to Gemini for transcription")
        words = transcribe_with_gemini(video_path)
        if words:
            srt = words_to_srt(words)
            return {"words": words, "srt": srt, "method": "gemini"}
    except Exception as e:
        logger.warning(f"Gemini fallback also failed: {e}")

    raise RuntimeError("Both Whisper and Gemini transcription failed")


def burn_subtitles(video_path: str, srt_content: str, output_path: str,
                   font_size: int = 0, font_color: str = "white",
                   outline_color: str = "black", outline_width: int = 3,
                   position: str = "bottom") -> str:
    """
    Burn SRT subtitles into video using FFmpeg.

    Args:
        video_path: Input video
        srt_content: SRT subtitle string
        output_path: Output video path
        font_size: Font size (0 = auto based on video height)
        font_color: Subtitle text color
        outline_color: Outline/border color
        outline_width: Outline thickness
        position: "bottom", "center", or "top"

    Returns:
        Path to output video
    """
    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())

    # Write SRT to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix=".srt", delete=False, encoding='utf-8') as tmp:
        tmp.write(srt_content)
        srt_path = tmp.name

    try:
        # Get video dimensions for auto font size
        w, h = _get_video_dimensions(video_path)
        if font_size <= 0:
            font_size = max(16, int(h * 0.045))

        # Position mapping
        margin_v = int(h * 0.06)
        if position == "top":
            alignment = 8  # ASS alignment: top-center
        elif position == "center":
            alignment = 5  # center
        else:
            alignment = 2  # bottom-center (default)

        # Escape path for FFmpeg filter (Windows backslashes)
        srt_escaped = srt_path.replace('\\', '/').replace(':', '\\:')

        # Build subtitle filter with styling
        sub_filter = (
            f"subtitles='{srt_escaped}':"
            f"force_style='FontSize={font_size},"
            f"PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,"
            f"BorderStyle=1,"
            f"Outline={outline_width},"
            f"Shadow=1,"
            f"Alignment={alignment},"
            f"MarginV={margin_v},"
            f"FontName=Arial'"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", sub_filter,
            "-c:a", "copy",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            output_path
        ]

        logger.info(f"Burning subtitles into video: {output_path}")
        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8', errors='replace'
        )
        if result.returncode != 0:
            logger.error(f"FFmpeg subtitle burn failed: {result.stderr}")
            raise RuntimeError(f"FFmpeg error: {result.stderr[-500:]}")

        logger.info(f"Subtitled video saved: {output_path}")
        return output_path
    finally:
        if os.path.exists(srt_path):
            os.unlink(srt_path)


def burn_text_overlay(video_path: str, output_path: str,
                      text_blocks: list) -> str:
    """
    Burn text overlays onto video at specific positions with per-word colors.

    Args:
        video_path: Input video
        output_path: Output video path
        text_blocks: List of text overlay specs:
            [{
                "text": "YOUR HEADLINE",
                "x": 0.5,  # normalized 0-1 (center)
                "y": 0.3,  # normalized 0-1
                "font_size": 48,
                "words": [
                    {"word": "YOUR", "color": "#CAFF00"},
                    {"word": "HEADLINE", "color": "#FFFFFF"}
                ]
            }]

    Returns:
        Path to output video
    """
    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())

    w, h = _get_video_dimensions(video_path)
    font_path = _get_font_path()
    font_param = f"fontfile='{font_path.replace(chr(92), '/').replace(':', chr(92)+':')}'" if font_path else "font=Arial"

    # Build drawtext filters for each word in each block
    filters = []
    for block in text_blocks:
        bx = block.get("x", 0.5)
        by = block.get("y", 0.5)
        font_size = block.get("font_size", 48)
        word_specs = block.get("words", [])

        if not word_specs:
            # Single-color text
            text = block.get("text", "")
            color = block.get("color", "white")
            px = int(bx * w)
            py = int(by * h)
            escaped = text.replace("'", "\\'").replace(":", "\\:")
            filters.append(
                f"drawtext=text='{escaped}':"
                f"x={px}-(text_w/2):y={py}-(text_h/2):"
                f"fontsize={font_size}:fontcolor={color}:"
                f"borderw=3:bordercolor=black:"
                f"{font_param}"
            )
        else:
            # Per-word coloring: lay out words horizontally
            full_text = " ".join(ws["word"] for ws in word_specs)
            total_chars = len(full_text)
            px_start = int(bx * w)
            py = int(by * h)

            # Approximate char width
            char_w = font_size * 0.55
            total_w = total_chars * char_w
            start_x = px_start - total_w / 2

            offset = 0
            for ws in word_specs:
                word = ws["word"]
                color = ws.get("color", "white")
                wx = int(start_x + offset * char_w)
                escaped = word.replace("'", "\\'").replace(":", "\\:")
                filters.append(
                    f"drawtext=text='{escaped}':"
                    f"x={wx}:y={py}-(text_h/2):"
                    f"fontsize={font_size}:fontcolor={color}:"
                    f"borderw=3:bordercolor=black:"
                    f"{font_param}"
                )
                offset += len(word) + 1  # +1 for space

    if not filters:
        raise ValueError("No text overlays provided")

    vf = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:a", "copy",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        output_path
    ]

    logger.info(f"Burning text overlay: {len(filters)} drawtext filters")
    result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        logger.error(f"FFmpeg overlay failed: {result.stderr}")
        raise RuntimeError(f"FFmpeg error: {result.stderr[-500:]}")

    logger.info(f"Overlay video saved: {output_path}")
    return output_path
