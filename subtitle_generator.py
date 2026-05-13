"""
Subtitle Generator Module
Generates SRT subtitles from video using Whisper (local) with Gemini fallback.
Burns subtitles into video using FFmpeg.
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _ffmpeg_font_setup() -> dict:
    """
    Set up a portable font for FFmpeg drawtext on Windows.
    Copies a system font to a temp dir so we can reference it without C: colon.
    Returns dict with 'env' (subprocess env) and 'font_param' (filter string).
    """
    setup_dir = os.path.join(tempfile.gettempdir(), "ffmpeg_fonts")
    os.makedirs(setup_dir, exist_ok=True)

    env = os.environ.copy()

    if platform.system() == "Windows":
        # 1. Create valid fontconfig config with forward slashes
        fc_conf = os.path.join(setup_dir, "fonts.conf")
        fonts_dir_fwd = "C:/Windows/Fonts"
        cache_dir_fwd = setup_dir.replace("\\", "/")
        if not os.path.isfile(fc_conf):
            with open(fc_conf, "w", encoding="utf-8") as f:
                f.write(
                    '<?xml version="1.0"?>\n'
                    '<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">\n'
                    '<fontconfig>\n'
                    f'  <dir>{fonts_dir_fwd}</dir>\n'
                    f'  <cachedir>{cache_dir_fwd}</cachedir>\n'
                    '</fontconfig>\n'
                )
        env["FONTCONFIG_FILE"] = fc_conf
        env["FONTCONFIG_PATH"] = setup_dir

        # 2. Copy a font file to the temp dir so we can use a colon-free path
        font_dst = os.path.join(setup_dir, "font.ttf")
        if not os.path.isfile(font_dst):
            for src in [
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/calibri.ttf",
            ]:
                if os.path.isfile(src):
                    shutil.copy2(src, font_dst)
                    break

        if os.path.isfile(font_dst):
            # Use the copied font — path has no colon issues in filter syntax
            # The font is in the cwd we'll set for subprocess, so just use filename
            font_param = f"fontfile=font.ttf"
        else:
            font_param = "font=Arial"
    else:
        # Linux — direct path, no colon issues
        for p in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        ]:
            if os.path.isfile(p):
                font_param = f"fontfile={p}"
                break
        else:
            font_param = "font=Arial"

    return {"env": env, "font_param": font_param, "cwd": setup_dir}


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

        # Escape SRT path — forward slashes, escape colons for filter syntax
        srt_escaped = srt_path.replace('\\', '/').replace(':', '\\:')
        font_setup = _ffmpeg_font_setup()

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
            "-preset", "veryfast",
            "-threads", "0",
            "-crf", "23",
            output_path
        ]

        logger.info(f"Burning subtitles into video: {output_path}")
        logger.info(f"Subtitle filter: {sub_filter}")
        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8', errors='replace',
            env=font_setup["env"],
            cwd=font_setup["cwd"],
        )
        if result.returncode != 0:
            logger.error(f"FFmpeg subtitle burn FULL stderr:\n{result.stderr}")
            raise RuntimeError(f"FFmpeg subtitle burn failed (code {result.returncode}). "
                               f"Check server logs for full stderr.")

        logger.info(f"Subtitled video saved: {output_path}")
        return output_path
    finally:
        if os.path.exists(srt_path):
            os.unlink(srt_path)


def burn_text_overlay(video_path: str, output_path: str,
                      text_blocks: list) -> str:
    """
    Burn text overlays onto video at specific positions with per-word colors.
    Uses -filter_script to avoid all Windows command-line escaping issues.

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
    font_setup = _ffmpeg_font_setup()
    font_param = font_setup["font_param"]

    def _esc(txt: str) -> str:
        """Escape text for FFmpeg drawtext — backslash-escape special chars."""
        for ch in ("\\", "'", ":", ";", "[", "]", ","):
            txt = txt.replace(ch, "\\" + ch)
        return txt

    # Build drawtext filters for each word in each block
    filters = []
    for block in text_blocks:
        bx = block.get("x", 0.5)
        by = block.get("y", 0.5)
        font_size = block.get("font_size", 48)
        word_specs = block.get("words", [])

        if not word_specs:
            text = block.get("text", "")
            color = block.get("color", "white")
            px = int(bx * w)
            py = int(by * h)
            escaped = _esc(text)
            filters.append(
                f"drawtext=text='{escaped}':"
                f"x={px}-(text_w/2):y={py}-(text_h/2):"
                f"fontsize={font_size}:fontcolor='{color}':"
                f"borderw=3:bordercolor=black:"
                f"{font_param}"
            )
        else:
            full_text = " ".join(ws["word"] for ws in word_specs)
            total_chars = len(full_text)
            px_start = int(bx * w)
            py = int(by * h)

            char_w = font_size * 0.55
            total_w = total_chars * char_w
            start_x = px_start - total_w / 2

            offset = 0
            for ws in word_specs:
                word = ws["word"]
                color = ws.get("color", "white")
                wx = int(start_x + offset * char_w)
                escaped = _esc(word)
                filters.append(
                    f"drawtext=text='{escaped}':"
                    f"x={wx}:y={py}-(text_h/2):"
                    f"fontsize={font_size}:fontcolor='{color}':"
                    f"borderw=3:bordercolor=black:"
                    f"{font_param}"
                )
                offset += len(word) + 1

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
    logger.info(f"VF filter: {vf}")
    logger.info(f"Font param: {font_param}, cwd: {font_setup['cwd']}")
    logger.info(f"Full cmd: {cmd}")
    result = subprocess.run(
        cmd, capture_output=True, encoding='utf-8', errors='replace',
        env=font_setup["env"],
        cwd=font_setup["cwd"],   # run FFmpeg from the font dir so fontfile=font.ttf resolves
    )

    if result.returncode != 0:
        logger.error(f"FFmpeg overlay FULL stderr:\n{result.stderr}")
        logger.error(f"FFmpeg overlay FULL stdout:\n{result.stdout}")
        raise RuntimeError(f"FFmpeg overlay failed (code {result.returncode}). "
                           f"Check server logs for full stderr.")

    logger.info(f"Overlay video saved: {output_path}")
    return output_path
