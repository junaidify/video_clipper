"""
TTS Engine Module
Converts commentary scripts to speech using Edge TTS with SSML
emotional enhancement — prosody contours, strategic pauses, emphasis,
and mstts:express-as voice styles for supported voices.
Higher audio quality output (320kbps, 48kHz).
"""
import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import html as html_lib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Audio quality settings ───
AUDIO_BITRATE = "320k"   # High quality
AUDIO_SAMPLE_RATE = "48000"  # Studio quality

# ─── Voice Style Map ───
# Voices that support mstts:express-as styles.
# style_name → best narration/motivation style for that voice.
# styledegree: 0.01-2.0 (higher = more intense emotion)
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

# ─── Emphasis Words ───
# Words/phrases that get <emphasis level="strong"> in SSML
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

# ─── Voice Presets ───
VOICE_PRESETS = {
    # ── Deep & Motivational (powerful, commanding) ──
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

    # ── English Narration (with emotional styles) ──
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

    # ── Indian English (Deep & Heavy) ──
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

    # ── Hindi (Deep & Heavy) ──
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

    # ── Tamil ──
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

    # ── Telugu ──
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

    # ── Bengali ──
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

    # ── Gujarati ──
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

    # ── Marathi ──
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

    # ── Kannada ──
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

    # ── Punjabi ──
    "gur_hero": {
        "id": "pa-IN-GurdeepNeural",
        "label": "Gurdeep (Punjabi Deep Male)",
        "gender": "male", "style": "motivation",
        "category": "punjabi",
    },

    # ── Urdu ──
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


# ─── SSML Builder ───

def _enhance_text(text: str) -> str:
    """
    Enhance plain text for more natural TTS delivery using ONLY punctuation.
    NO XML/SSML tags — edge_tts handles all markup internally.

    Techniques:
    - Ellipsis pauses between sentences for breathing room
    - Em-dashes for dramatic emphasis
    - Capitalize power words for subtle stress
    - Commas before key phrases to force micro-pauses
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return text

    enhanced = []
    for i, sentence in enumerate(sentences):
        # Capitalize power words for subtle emphasis
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

    # Join with ellipsis pauses — forces natural breathing between sentences
    return " ... ".join(enhanced)


@dataclass
class TTSResult:
    """Result of TTS synthesis."""
    audio_path: str         # path to generated audio file
    duration: float         # audio duration in seconds
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


def get_available_voices() -> list:
    """Return list of preset voices for the UI, grouped by category."""
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
    """
    Safely run an async coroutine from a sync context (including threads).
    Handles the event loop lifecycle properly to avoid conflicts with
    Flask's dev server or other running loops.
    """
    try:
        # Check if there's already a running loop (e.g., inside Flask debug mode)
        loop = asyncio.get_running_loop()
        # We're inside an existing loop — can't use run_until_complete.
        # Spin up a new thread with its own loop instead.
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
        t.join(timeout=300)  # 5 minute max
        if error[0]:
            raise error[0]
        return result[0]
    except RuntimeError:
        # No running loop — safe to create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


async def _synthesize_segment(text: str, voice_id: str, output_path: str,
                               rate: str = "+0%", pitch: str = "+0Hz",
                               narration_style: str = "narration") -> dict:
    """
    Synthesize a single text segment to audio file using Edge TTS.
    Uses PURE PLAIN TEXT only — no XML/SSML tags whatsoever.
    Naturalness comes from punctuation-based text enhancement.
    Returns dict with path and duration, or error.
    """
    import edge_tts

    try:
        # Enhance text with punctuation (NO XML tags)
        enhanced = _enhance_text(text)
        logger.debug(f"Enhanced text: {enhanced[:200]}...")

        # Plain text + rate/pitch — nothing else
        communicate = edge_tts.Communicate(enhanced, voice_id, rate=rate, pitch=pitch)
        await communicate.save(output_path)

        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            return {"path": None, "error": "Edge TTS produced empty file"}

        duration = _get_audio_duration(output_path)
        return {"path": output_path, "duration": duration}

    except Exception as e:
        logger.error(f"Edge TTS synthesis error: {type(e).__name__}: {e}")
        return {"path": None, "error": f"{type(e).__name__}: {e}"}


async def _synthesize_all_segments(segments: list, voice_id: str,
                                    output_dir: str,
                                    rate: str = "+0%",
                                    pitch: str = "+0Hz",
                                    narration_style: str = "narration") -> list:
    """
    Synthesize all commentary segments to individual audio files with SSML.
    Each segment is processed sequentially (Edge TTS doesn't support
    concurrent connections reliably).
    """
    results = []
    first_error = None

    for i, seg in enumerate(segments):
        seg_path = os.path.join(output_dir, f"seg_{i:03d}.mp3")
        text = seg.get("text", "").strip()
        if not text:
            logger.warning(f"  Segment {i}: empty text, skipping")
            continue

        logger.info(f"  TTS segment {i}/{len(segments)-1}: \"{text[:60]}...\"")

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
            logger.info(f"    → {result['duration']:.1f}s audio for [{seg['start']:.1f}s-{seg['end']:.1f}s]")
        else:
            err = result.get("error", "Unknown error")
            if not first_error:
                first_error = err
            logger.error(f"    → FAILED: {err}")
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


def synthesize_commentary(commentary_segments: list,
                          output_dir: str,
                          voice_key: str = DEFAULT_VOICE,
                          rate: str = "+0%",
                          pitch: str = "+0Hz",
                          video_duration: float = 0,
                          narration_style: str = "narration") -> TTSResult:
    """
    Synthesize full commentary script to a single merged audio file
    with SSML emotional enhancement.

    Args:
        commentary_segments: List of dicts with 'start', 'end', 'text', 'pause_after'
        output_dir: Directory to save output files
        voice_key: Voice preset key from VOICE_PRESETS
        rate: Speech rate adjustment (e.g., "+10%", "-5%")
        pitch: Pitch adjustment (e.g., "+2Hz", "-1Hz")
        video_duration: Target video duration for padding
        narration_style: Style hint ('narration', 'motivation', 'sports', 'summary')

    Returns:
        TTSResult with path to the final merged audio
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolve voice ID
    voice_info = VOICE_PRESETS.get(voice_key, VOICE_PRESETS[DEFAULT_VOICE])
    voice_id = voice_info["id"]

    # Use the voice's own style category if narration_style is generic
    effective_style = narration_style
    if narration_style == "narration" and voice_info.get("style") == "motivation":
        effective_style = "motivation"

    logger.info(
        f"Synthesizing {len(commentary_segments)} segments with voice: "
        f"{voice_info['label']} ({voice_id}), style: {effective_style}"
    )

    # Synthesize all segments using safe async runner
    try:
        seg_results = _run_async(
            _synthesize_all_segments(
                commentary_segments, voice_id, output_dir, rate, pitch,
                narration_style=effective_style
            )
        )
    except Exception as e:
        logger.error(f"TTS async execution failed: {type(e).__name__}: {e}", exc_info=True)
        return TTSResult(
            audio_path="",
            duration=0,
            voice_id=voice_id,
            success=False,
            error=f"TTS engine error: {type(e).__name__}: {e}",
        )

    # Check for failures
    successful = [r for r in seg_results if r.get("path")]
    failed = [r for r in seg_results if not r.get("path")]

    if not successful:
        # Collect all error messages for debugging
        errors = [r.get("error", "unknown") for r in failed]
        unique_errors = list(set(errors))
        error_msg = f"All {len(failed)} TTS segments failed. Errors: {'; '.join(unique_errors[:3])}"
        logger.error(error_msg)
        return TTSResult(
            audio_path="",
            duration=0,
            voice_id=voice_id,
            success=False,
            error=error_msg,
        )

    if failed:
        logger.warning(f"{len(failed)}/{len(seg_results)} segments failed, continuing with {len(successful)} successful")

    # Merge segments into a single timeline-aligned audio file
    merged_path = os.path.join(output_dir, "commentary_merged.mp3")
    total_duration = _merge_segments_to_timeline(
        seg_results, merged_path, video_duration
    )

    logger.info(f"Commentary audio: {total_duration:.1f}s, saved to {merged_path}")

    return TTSResult(
        audio_path=merged_path,
        duration=total_duration,
        voice_id=voice_id,
        success=True,
    )


def _merge_segments_to_timeline(seg_results: list, output_path: str,
                                 video_duration: float) -> float:
    """
    Merge individual TTS audio segments into a single file,
    placing each at its target start time with silence padding.
    Uses FFmpeg's adelay + amix filters.
    """
    valid_segs = [r for r in seg_results if r.get("path") and os.path.isfile(r["path"])]
    if not valid_segs:
        return 0.0

    if len(valid_segs) == 1:
        # Single segment — just pad with silence to match timeline
        seg = valid_segs[0]
        delay_ms = int(seg["target_start"] * 1000)
        target_dur = max(video_duration, seg["target_start"] + seg["audio_duration"] + 1)

        # Volume boost: Edge TTS output is often quiet, boost by 2.5x and limit
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

    # Multiple segments — build complex filter
    # Key fix: amix divides volume by input count by default.
    # We add normalize=0 to prevent that, plus a volume boost + limiter.
    n = len(valid_segs)
    inputs = []
    filter_parts = []

    for i, seg in enumerate(valid_segs):
        inputs.extend(["-i", seg["path"]])
        delay_ms = int(seg["target_start"] * 1000)
        # Boost each segment before mixing
        filter_parts.append(f"[{i}:a]volume=2.5,adelay={delay_ms}|{delay_ms},apad=whole_dur={video_duration}[a{i}]")

    # Mix all delayed streams — normalize=0 prevents volume division
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
        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8',
            errors='replace', timeout=120
        )
        if result.returncode != 0:
            logger.error(f"FFmpeg merge failed: {result.stderr[-300:]}")
            # Fallback: simple concatenation
            return _simple_concat_fallback(valid_segs, output_path, video_duration)

        return _get_audio_duration(output_path)

    except Exception as e:
        logger.error(f"Merge error: {e}")
        return _simple_concat_fallback(valid_segs, output_path, video_duration)


def _simple_concat_fallback(seg_results: list, output_path: str,
                             video_duration: float) -> float:
    """Fallback: concatenate segments sequentially if timeline merge fails."""
    # Build concat file
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
        subprocess.run(cmd, capture_output=True, encoding='utf-8',
                       errors='replace', timeout=60)
    except Exception:
        pass

    # Cleanup
    try:
        os.remove(concat_file)
    except Exception:
        pass

    return _get_audio_duration(output_path)


def _get_audio_duration(path: str) -> float:
    """Get audio file duration via ffprobe."""
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
