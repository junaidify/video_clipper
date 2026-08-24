"""
Subtitle Generator Module
Generates word-level SRT subtitles from audio using local Whisper with Gemini API fallback.
Burns subtitles and text overlays into video using FFmpeg and Pillow.
"""
import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional, List

warnings.filterwarnings("ignore", message=".*Triton.*falling back.*")
warnings.filterwarnings("ignore", message=".*Failed to launch Triton kernels.*")

logger = logging.getLogger(__name__)

# --- Devanagari to Roman (Hinglish) transliteration map ---
_DEVA_TO_ROMAN = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ऋ': 'ri',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'n',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
    'ष': 'sh', 'स': 's', 'ह': 'h',
    'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'फ़': 'f', 'ड़': 'r', 'ढ़': 'rh',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ृ': 'ri',
    '्': '', 'ं': 'n', 'ँ': 'n', 'ः': 'h', 'ऽ': '',
    '।': '.', '॥': '.',
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}


def _devanagari_to_roman(text: str) -> str:
    result = []
    i = 0
    while i < len(text):
        if i + 1 < len(text) and text[i:i+2] in _DEVA_TO_ROMAN:
            result.append(_DEVA_TO_ROMAN[text[i:i+2]])
            i += 2
        elif text[i] in _DEVA_TO_ROMAN:
            result.append(_DEVA_TO_ROMAN[text[i]])
            i += 1
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def _has_devanagari(text: str) -> bool:
    return any(0x0900 <= ord(c) <= 0x097F for c in text)


def _to_hinglish(text: str) -> str:
    if not _has_devanagari(text):
        return text
    return _devanagari_to_roman(text)


def _ffmpeg_font_setup() -> dict:
    """Set up a portable font config for FFmpeg drawtext on Windows/Linux."""
    setup_dir = os.path.join(tempfile.gettempdir(), "ffmpeg_fonts")
    os.makedirs(setup_dir, exist_ok=True)

    env = os.environ.copy()

    if platform.system() == "Windows":
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
            font_param = "fontfile=font.ttf"
        else:
            font_param = "font=Arial"
    else:
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


def transcribe_with_whisper(video_path: str, model_size: str = "base", language: Optional[str] = None) -> List[dict]:
    """Transcribe video using local Whisper and return word-level timestamps."""
    try:
        import whisper
    except ImportError:
        raise ImportError("Whisper is not installed. Run: pip install openai-whisper")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path
        ]
        subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', check=True)

        from video_clipper.clipping.transcriber import resolve_device
        model = whisper.load_model(model_size, device=resolve_device("auto"))
        whisper_lang = 'hi' if language in ('hinglish', 'hi') else (language if language else None)
        logger.info(f"Whisper subtitle language: {whisper_lang} (requested: {language})")

        options = {"word_timestamps": True, "verbose": False}
        if whisper_lang:
            options["language"] = whisper_lang

        result = model.transcribe(audio_path, **options)

        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                word_text = w["word"].strip()
                if language in ('hinglish', 'hi') and _has_devanagari(word_text):
                    word_text = _to_hinglish(word_text)
                words.append({
                    "word": word_text,
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                })

        logger.info(f"Whisper transcribed {len(words)} words from {video_path}")
        return words
    finally:
        if os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass


def transcribe_with_gemini(video_path: str, language: Optional[str] = None) -> List[dict]:
    """Fallback: transcribe using Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Install google-generativeai: pip install google-generativeai")

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
        audio_file = genai.upload_file(audio_path, mime_type="audio/mpeg")

        if language in ("hinglish", "hi"):
            lang_instruction = "Transcribe this Hindi audio in ROMAN/LATIN script (Hinglish). Example: main kitne dinon tak rahunga. Do NOT use Devanagari or Arabic script."
        else:
            lang_instruction = "Transcribe this English audio accurately."

        prompt = f"""{lang_instruction}
Return ONLY a JSON array, no markdown, no explanation.
Each element: {{"word": "the_word", "start": seconds_float, "end": seconds_float}}
Group into natural phrases of 4-8 words for subtitle display.
Example: [{{"word": "Hello world", "start": 0.0, "end": 0.8}}, ...]"""

        response = model.generate_content([prompt, audio_file])
        text = response.text.strip()

        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        words = json.loads(text)
        logger.info(f"Gemini transcribed {len(words)} subtitle segments from {video_path}")
        return words
    finally:
        if os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass


def words_to_srt(
    words: List[dict],
    max_words_per_line: int = 6,
    max_chars_per_line: int = 40,
    max_duration: float = 4.0,
) -> str:
    """Convert word-level timestamps to SRT subtitle format."""
    if not words:
        return ""

    subtitles = []
    current_words = []
    current_start = None

    for w in words:
        if current_start is None:
            current_start = w["start"]

        test_line = " ".join(current_words + [w["word"]])
        elapsed = w["end"] - current_start

        if current_words and (
            len(current_words) >= max_words_per_line
            or len(test_line) > max_chars_per_line
            or elapsed >= max_duration
        ):
            subtitles.append({
                "index": len(subtitles) + 1,
                "start": current_start,
                "end": w["start"],
                "text": " ".join(current_words),
            })
            current_words = [w["word"]]
            current_start = w["start"]
        else:
            current_words.append(w["word"])

    if current_words:
        subtitles.append({
            "index": len(subtitles) + 1,
            "start": current_start,
            "end": words[-1]["end"],
            "text": " ".join(current_words),
        })

    srt_lines = []
    for sub in subtitles:
        srt_lines.append(str(sub["index"]))
        srt_lines.append(f"{_format_srt_time(sub['start'])} --> {_format_srt_time(sub['end'])}")
        srt_lines.append(sub["text"])
        srt_lines.append("")

    return "\n".join(srt_lines)


def generate_subtitles(video_path: str, model_size: str = "base", language: Optional[str] = None) -> dict:
    """
    Generate subtitles for a video file. Tries Whisper locally, falls back to Gemini.

    Returns:
        {"words": [...], "srt": "...", "method": "whisper"|"gemini"}
    """
    video_path = str(Path(video_path).resolve())

    # Try Whisper first
    try:
        logger.info(f"Attempting Whisper transcription (model={model_size})")
        words = transcribe_with_whisper(video_path, model_size, language=language)
        if words:
            srt = words_to_srt(words)
            return {"words": words, "srt": srt, "method": "whisper"}
    except Exception as e:
        logger.warning(f"Whisper failed: {e}")

    # Fallback to Gemini
    try:
        logger.info("Falling back to Gemini for transcription")
        words = transcribe_with_gemini(video_path, language=language)
        if words:
            srt = words_to_srt(words)
            return {"words": words, "srt": srt, "method": "gemini"}
    except Exception as e:
        logger.warning(f"Gemini fallback also failed: {e}")

    raise RuntimeError("Both Whisper and Gemini transcription failed")


def burn_subtitles(
    video_path: str,
    srt_content: str,
    output_path: str,
    font_size: int = 0,
    font_color: str = "white",
    outline_color: str = "black",
    outline_width: int = 3,
    position: str = "bottom",
) -> str:
    """Burn SRT subtitles into video using FFmpeg."""
    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())

    with tempfile.NamedTemporaryFile(mode='w', suffix=".srt", delete=False, encoding='utf-8') as tmp:
        tmp.write(srt_content)
        srt_path = tmp.name

    try:
        w, h = _get_video_dimensions(video_path)
        is_vertical = h > w

        if font_size <= 0:
            if is_vertical:
                font_size = max(23, int(w * 0.04) + 3)
            else:
                font_size = max(21, int(h * 0.035) + 3)

        margin_l = int(w * 0.05)
        margin_r = margin_l

        if position == "top":
            alignment = 8
            margin_v = int(h * 0.05)
        elif position == "center":
            alignment = 5
            margin_v = 0
        else:
            alignment = 2
            if is_vertical:
                margin_v = int(h * 0.18)
            else:
                margin_v = int(h * 0.05)

        srt_escaped = srt_path.replace('\\', '/').replace(':', '\\:')
        font_setup = _ffmpeg_font_setup()

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
            f"MarginL={margin_l},"
            f"MarginR={margin_r},"
            f"WrapStyle=1,"
            f"PlayResX={w},"
            f"PlayResY={h},"
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
        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8', errors='replace',
            env=font_setup["env"],
            cwd=font_setup["cwd"],
        )
        if result.returncode != 0:
            logger.error(f"FFmpeg subtitle burn failed: {result.stderr}")
            raise RuntimeError(f"FFmpeg subtitle burn failed (code {result.returncode}).")

        return output_path
    finally:
        if os.path.exists(srt_path):
            os.unlink(srt_path)


def burn_text_overlay(video_path: str, output_path: str, text_blocks: List[dict]) -> str:
    """Burn text overlays onto video using Pillow for text rendering + FFmpeg for compositing."""
    from PIL import Image, ImageDraw, ImageFont

    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())

    w, h = _get_video_dimensions(video_path)

    font_path = None
    for candidate in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Nirmala.ttf",
        "C:/Windows/Fonts/NirmalaB.ttf",
        "C:/Windows/Fonts/mangal.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ]:
        if os.path.isfile(candidate):
            font_path = candidate
            break

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for block in text_blocks:
        bx = block.get("x", 0.5)
        by = block.get("y", 0.5)
        font_size = block.get("font_size", 48)
        bg_opacity = block.get("bg_opacity", 0.0)
        word_specs = block.get("words", [])

        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        if not word_specs:
            text = block.get("text", "")
            color = block.get("color", "#FFFFFF")
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            px = int(bx * w) - tw // 2
            py = int(by * h) - th // 2

            if bg_opacity > 0:
                pad = 10
                bg_color = (0, 0, 0, int(255 * bg_opacity))
                draw.rounded_rectangle([px-pad, py-pad, px+tw+pad, py+th+pad], radius=8, fill=bg_color)

            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx*dx + dy*dy <= 9:
                        draw.text((px+dx, py+dy), text, font=font, fill=(0, 0, 0, 255))
            draw.text((px, py), text, font=font, fill=color)
        else:
            full_text = " ".join(ws.get("word", "") for ws in word_specs)
            bbox = draw.textbbox((0, 0), full_text, font=font)
            total_w = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            start_x = int(bx * w) - total_w // 2
            py = int(by * h) - th // 2

            if bg_opacity > 0:
                pad = 12
                bg_color = (0, 0, 0, int(255 * bg_opacity))
                draw.rounded_rectangle([start_x-pad, py-pad, start_x+total_w+pad, py+th+pad], radius=8, fill=bg_color)

            cx = start_x
            for ws in word_specs:
                word = ws.get("word", "")
                color_hex = ws.get("color", "#FFFFFF")
                c = color_hex.lstrip("#")
                rgb = tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) + (255,)

                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx*dx + dy*dy <= 9:
                            draw.text((cx+dx, py+dy), word, font=font, fill=(0, 0, 0, 255))
                draw.text((cx, py), word, font=font, fill=rgb)

                word_bbox = draw.textbbox((0, 0), word + " ", font=font)
                cx += word_bbox[2] - word_bbox[0]

    overlay_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    overlay.save(overlay_png)

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", overlay_png,
            "-filter_complex", "[0:v][1:v]overlay=0:0",
            "-c:a", "copy",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg overlay composite failed (code {result.returncode})")

        return output_path
    finally:
        if os.path.exists(overlay_png):
            os.unlink(overlay_png)
