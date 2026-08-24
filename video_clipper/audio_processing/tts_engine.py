"""
TTS Engine Module
Converts commentary and story scripts to natural speech using Microsoft Edge TTS.
Uses punctuation-based prosody enhancement, natural breathing pauses, and studio-grade 320kbps 48kHz audio.
"""
import asyncio
import html as html_lib
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

AUDIO_BITRATE = "320k"
AUDIO_SAMPLE_RATE = "48000"

VOICE_STYLES = {
    "en-US-AriaNeural": {
        "narration":   {"style": "narration-professional", "styledegree": "1.5"},
        "motivation":  {"style": "excited", "styledegree": "1.8"},
        "sports":      {"style": "excited", "styledegree": "2.0"},
        "summary":     {"style": "narration-professional", "styledegree": "1.2"},
    },
    "en-US-DavisNeural": {
        "narration":   {"style": "friendly", "styledegree": "1.3"},
        "motivation":  {"style": "excited", "styledegree": "1.8"},
        "sports":      {"style": "shouting", "styledegree": "1.5"},
        "summary":     {"style": "friendly", "styledegree": "1.0"},
    },
    "en-US-GuyNeural": {
        "narration":   {"style": "newscast", "styledegree": "1.4"},
        "motivation":  {"style": "excited", "styledegree": "1.8"},
        "sports":      {"style": "excited", "styledegree": "2.0"},
        "summary":     {"style": "newscast", "styledegree": "1.2"},
    },
    "en-US-JennyNeural": {
        "narration":   {"style": "narration-professional", "styledegree": "1.5"},
        "motivation":  {"style": "excited", "styledegree": "1.6"},
        "sports":      {"style": "excited", "styledegree": "2.0"},
        "summary":     {"style": "narration-professional", "styledegree": "1.2"},
    },
    "en-US-ChristopherNeural": {
        "narration":   {"style": "narration-professional", "styledegree": "1.5"},
        "motivation":  {"style": "excited", "styledegree": "1.6"},
        "sports":      {"style": "excited", "styledegree": "1.8"},
        "summary":     {"style": "narration-professional", "styledegree": "1.2"},
    },
}

EMPHASIS_WORDS = {
    "incredible", "amazing", "unbelievable", "extraordinary", "spectacular",
    "powerful", "unstoppable", "legendary", "epic", "massive", "insane",
    "critical", "crucial", "devastating", "dominant", "explosive", "fierce",
    "game-changing", "historic", "intense", "monumental", "phenomenal",
    "relentless", "savage", "shocking", "stunning", "supreme", "ultimate",
    "victory", "warrior", "champion", "never", "always", "everything",
    "nothing", "impossible", "greatest", "strongest", "fastest", "deadliest",
    "absolutely", "completely", "totally", "literally", "definitely",
}

VOICE_PRESETS = {
    "andrew": {
        "id": "en-US-AndrewMultilingualNeural",
        "label": "Andrew (Deep Motivational)",
        "gender": "male", "style": "motivation",
        "category": "deep",
    },
    "brian": {
        "id": "en-US-BrianMultilingualNeural",
        "label": "Brian (Deep Commanding)",
        "gender": "male", "style": "motivation",
        "category": "deep",
    },
    "christopher": {
        "id": "en-US-ChristopherNeural",
        "label": "Christopher (Authoritative)",
        "gender": "male", "style": "motivation",
        "category": "deep",
    },
    "roger": {
        "id": "en-US-RogerNeural",
        "label": "Roger (Deep Cinematic)",
        "gender": "male", "style": "motivation",
        "category": "deep",
    },
    "davis": {
        "id": "en-US-DavisNeural",
        "label": "Davis (Expressive Male)",
        "gender": "male", "style": "narration",
        "category": "narration",
    },
    "guy_narrator": {
        "id": "en-US-GuyNeural",
        "label": "Guy (Newscast Male)",
        "gender": "male", "style": "narration",
        "category": "narration",
    },
    "aria": {
        "id": "en-US-AriaNeural",
        "label": "Aria (Expressive Female)",
        "gender": "female", "style": "narration",
        "category": "narration",
    },
    "ryan": {
        "id": "en-GB-RyanNeural",
        "label": "Ryan (British Male)",
        "gender": "male", "style": "narration",
        "category": "narration",
    },
    "prabhat": {
        "id": "en-IN-PrabhatNeural",
        "label": "Prabhat (Indian Deep Male)",
        "gender": "male", "style": "motivation",
        "category": "indian",
    },
    "neerja": {
        "id": "en-IN-NeerjaNeural",
        "label": "Neerja (Indian Female)",
        "gender": "female", "style": "narration",
        "category": "indian",
    },
    "madhur": {
        "id": "hi-IN-MadhurNeural",
        "label": "Madhur (Hindi Deep Male)",
        "gender": "male", "style": "motivation",
        "category": "hindi",
    },
    "swara": {
        "id": "hi-IN-SwaraNeural",
        "label": "Swara (Hindi Female)",
        "gender": "female", "style": "narration",
        "category": "hindi",
    },
    "valluvar": {
        "id": "ta-IN-ValluvarNeural",
        "label": "Valluvar (Tamil Deep Male)",
        "gender": "male", "style": "motivation",
        "category": "tamil",
    },
    "pallavi_ta": {
        "id": "ta-IN-PallaviNeural",
        "label": "Pallavi (Tamil Female)",
        "gender": "female", "style": "narration",
        "category": "tamil",
    },
    "mohan": {
        "id": "te-IN-MohanNeural",
        "label": "Mohan (Telugu Deep Male)",
        "gender": "male", "style": "motivation",
        "category": "telugu",
    },
    "shruti": {
        "id": "te-IN-ShrutiNeural",
        "label": "Shruti (Telugu Female)",
        "gender": "female", "style": "narration",
        "category": "telugu",
    },
    "bashkar": {
        "id": "bn-IN-BashkarNeural",
        "label": "Bashkar (Bengali Deep Male)",
        "gender": "male", "style": "motivation",
        "category": "bengali",
    },
    "tanishaa": {
        "id": "bn-IN-TanishaaNeural",
        "label": "Tanishaa (Bengali Female)",
        "gender": "female", "style": "narration",
        "category": "bengali",
    },
    "niranjan": {
        "id": "gu-IN-NiranjanNeural",
        "label": "Niranjan (Gujarati Deep Male)",
        "gender": "male", "style": "motivation",
        "category": "gujarati",
    },
    "dhwani": {
        "id": "gu-IN-DhwaniNeural",
        "label": "Dhwani (Gujarati Female)",
        "gender": "female", "style": "narration",
        "category": "gujarati",
    },
    "manohar": {
        "id": "mr-IN-ManoharNeural",
        "label": "Manohar (Marathi Deep Male)",
        "gender": "male", "style": "motivation",
        "category": "marathi",
    },
    "aarohi": {
        "id": "mr-IN-AarohiNeural",
        "label": "Aarohi (Marathi Female)",
        "gender": "female", "style": "narration",
        "category": "marathi",
    },
    "gagan": {
        "id": "kn-IN-GaganNeural",
        "label": "Gagan (Kannada Deep Male)",
        "gender": "male", "style": "motivation",
        "category": "kannada",
    },
    "sapna": {
        "id": "kn-IN-SapnaNeural",
        "label": "Sapna (Kannada Female)",
        "gender": "female", "style": "narration",
        "category": "kannada",
    },
    "gur_hero": {
        "id": "pa-IN-GurdeepNeural",
        "label": "Gurdeep (Punjabi Deep Male)",
        "gender": "male", "style": "motivation",
        "category": "punjabi",
    },
    "asad": {
        "id": "ur-PK-AsadNeural",
        "label": "Asad (Urdu Male)",
        "gender": "male", "style": "narration",
        "category": "urdu",
    },
    "uzma": {
        "id": "ur-PK-UzmaNeural",
        "label": "Uzma (Urdu Female)",
        "gender": "female", "style": "narration",
        "category": "urdu",
    },
}

DEFAULT_VOICE = "andrew"


def _enhance_text(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return text

    enhanced = []
    for sentence in sentences:
        words = sentence.split()
        processed = []
        for word in words:
            clean = re.sub(r'[^a-zA-Z]', '', word).lower()
            if clean in EMPHASIS_WORDS and not word.isupper():
                processed.append(word.upper())
            else:
                processed.append(word)

        enhanced_sentence = " ".join(processed)
        enhanced.append(enhanced_sentence)

    return " ... ".join(enhanced)


@dataclass
class TTSResult:
    """Result of TTS voice synthesis."""
    audio_path: str
    duration: float
    voice_id: str
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "audio_path": self.audio_path,
            "duration": self.duration,
            "voice_id": self.voice_id,
            "success": self.success,
            "error": self.error,
        }


def get_available_voices() -> List[dict]:
    """Return list of preset voices grouped by category for the UI."""
    voices = []
    for key, info in VOICE_PRESETS.items():
        voices.append({
            "key": key,
            "id": info["id"],
            "label": info["label"],
            "gender": info["gender"],
            "style": info["style"],
            "category": info.get("category", "other"),
        })
    return voices


def _run_async(coro):
    """Safely execute an async coroutine from synchronous contexts."""
    try:
        loop = asyncio.get_running_loop()
        result = [None]
        error = [None]

        def _thread_target():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result[0] = new_loop.run_until_complete(coro)
            except Exception as e:
                error[0] = e
            finally:
                new_loop.close()

        t = threading.Thread(target=_thread_target)
        t.start()
        t.join(timeout=300)
        if error[0]:
            raise error[0]
        return result[0]
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


async def _synthesize_segment(
    text: str,
    voice_id: str,
    output_path: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    narration_style: str = "narration",
) -> dict:
    """Synthesize a single text segment to audio file via Edge TTS."""
    import edge_tts

    try:
        enhanced = _enhance_text(text)
        communicate = edge_tts.Communicate(enhanced, voice_id, rate=rate, pitch=pitch)
        await communicate.save(output_path)

        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            return {"path": None, "error": "Edge TTS produced empty file"}

        duration = _get_audio_duration(output_path)
        return {"path": output_path, "duration": duration}

    except Exception as e:
        logger.error(f"Edge TTS synthesis error: {e}")
        return {"path": None, "error": str(e)}


async def _synthesize_all_segments(
    segments: list,
    voice_id: str,
    output_dir: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    narration_style: str = "narration",
) -> list:
    results = []
    for i, seg in enumerate(segments):
        seg_path = os.path.join(output_dir, f"seg_{i:03d}.mp3")
        text = seg.get("text", "").strip()
        if not text:
            continue

        result = await _synthesize_segment(
            text, voice_id, seg_path, rate, pitch,
            narration_style=narration_style
        )

        if result.get("path"):
            results.append({
                "index": i,
                "path": result["path"],
                "audio_duration": result["duration"],
                "target_start": seg["start"],
                "target_end": seg["end"],
                "text": text,
                "pause_after": seg.get("pause_after", 0.3),
            })
        else:
            err = result.get("error", "Unknown error")
            results.append({
                "index": i,
                "path": None,
                "audio_duration": 0,
                "target_start": seg["start"],
                "target_end": seg["end"],
                "text": text,
                "error": err,
            })

    return results


def synthesize_commentary(
    commentary_segments: list,
    output_dir: str,
    voice_key: str = DEFAULT_VOICE,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    video_duration: float = 0,
    narration_style: str = "narration",
) -> TTSResult:
    """Synthesize full commentary script to timeline-aligned audio file."""
    os.makedirs(output_dir, exist_ok=True)

    voice_info = VOICE_PRESETS.get(voice_key, VOICE_PRESETS[DEFAULT_VOICE])
    voice_id = voice_info["id"]

    effective_style = narration_style
    if narration_style == "narration" and voice_info.get("style") == "motivation":
        effective_style = "motivation"

    logger.info(
        f"Synthesizing {len(commentary_segments)} segments with voice: "
        f"{voice_info['label']} ({voice_id})"
    )

    try:
        seg_results = _run_async(
            _synthesize_all_segments(
                commentary_segments, voice_id, output_dir, rate, pitch,
                narration_style=effective_style
            )
        )
    except Exception as e:
        logger.error(f"TTS execution failed: {e}")
        return TTSResult("", 0, voice_id, False, f"TTS engine error: {e}")

    successful = [r for r in seg_results if r.get("path")]
    failed = [r for r in seg_results if not r.get("path")]

    if not successful:
        errors = [r.get("error", "unknown") for r in failed]
        unique_errors = list(set(errors))
        error_msg = f"All {len(failed)} TTS segments failed: {'; '.join(unique_errors[:3])}"
        return TTSResult("", 0, voice_id, False, error_msg)

    merged_path = os.path.join(output_dir, "commentary_merged.mp3")
    total_duration = _merge_segments_to_timeline(
        seg_results, merged_path, video_duration
    )

    return TTSResult(
        audio_path=merged_path,
        duration=total_duration,
        voice_id=voice_id,
        success=True,
    )


def _merge_segments_to_timeline(seg_results: list, output_path: str, video_duration: float) -> float:
    valid_segs = [r for r in seg_results if r.get("path") and os.path.isfile(r["path"])]
    if not valid_segs:
        return 0.0

    if len(valid_segs) == 1:
        seg = valid_segs[0]
        delay_ms = int(seg["target_start"] * 1000)
        target_dur = max(video_duration, seg["target_start"] + seg["audio_duration"] + 1)

        cmd = [
            "ffmpeg", "-y",
            "-i", seg["path"],
            "-af", f"adelay={delay_ms}|{delay_ms},volume=2.5,alimiter=limit=0.95,apad=whole_dur={target_dur}",
            "-t", str(target_dur),
            "-c:a", "libmp3lame", "-b:a", AUDIO_BITRATE, "-ar", AUDIO_SAMPLE_RATE,
            output_path
        ]
        subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=60)
        return _get_audio_duration(output_path)

    n = len(valid_segs)
    inputs = []
    filter_parts = []

    for i, seg in enumerate(valid_segs):
        inputs.extend(["-i", seg["path"]])
        delay_ms = int(seg["target_start"] * 1000)
        filter_parts.append(f"[{i}:a]volume=2.5,adelay={delay_ms}|{delay_ms},apad=whole_dur={video_duration}[a{i}]")

    mix_inputs = "".join(f"[a{i}]" for i in range(n))
    filter_parts.append(
        f"{mix_inputs}amix=inputs={n}:duration=longest:dropout_transition=0:normalize=0,alimiter=limit=0.95[out]"
    )

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-t", str(video_duration),
        "-c:a", "libmp3lame", "-b:a", AUDIO_BITRATE, "-ar", AUDIO_SAMPLE_RATE,
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=120)
        if result.returncode != 0:
            return _simple_concat_fallback(valid_segs, output_path, video_duration)
        return _get_audio_duration(output_path)
    except Exception:
        return _simple_concat_fallback(valid_segs, output_path, video_duration)


def _simple_concat_fallback(seg_results: list, output_path: str, video_duration: float) -> float:
    concat_dir = os.path.dirname(output_path)
    concat_file = os.path.join(concat_dir, "concat_list.txt")

    with open(concat_file, "w") as f:
        for seg in seg_results:
            if seg.get("path") and os.path.isfile(seg["path"]):
                safe_path = seg["path"].replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c:a", "libmp3lame", "-b:a", AUDIO_BITRATE, "-ar", AUDIO_SAMPLE_RATE,
        output_path
    ]

    try:
        subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=60)
    except Exception:
        pass

    try:
        os.remove(concat_file)
    except Exception:
        pass

    return _get_audio_duration(output_path)


def _get_audio_duration(path: str) -> float:
    if not os.path.isfile(path):
        return 0.0
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", path
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        info = json.loads(r.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0
