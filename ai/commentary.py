"""
Commentary Generator Module
Analyzes video transcript and generates narration/voiceover scripts.
Uses NVIDIA NIM (OpenAI-compatible) to produce timed commentary segments.
"""
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CommentarySegment:
    """A single timed commentary segment."""
    start: float        # seconds
    end: float          # seconds
    text: str           # narration text
    pause_after: float = 0.3  # pause in seconds after this segment

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "pause_after": self.pause_after,
        }


@dataclass
class CommentaryScript:
    """Full commentary script with timed segments."""
    segments: list          # List[CommentarySegment]
    full_text: str          # complete narration text
    video_duration: float   # source video duration
    style: str = "narration"

    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "full_text": self.full_text,
            "video_duration": self.video_duration,
            "style": self.style,
            "total_segments": len(self.segments),
        }


def generate_commentary(transcript_segments: list,
                        video_duration: float,
                        style: str = "narration",
                        custom_prompt: str = "",
                        nvidia_api_key: str = "",
                        nvidia_model: str = "meta/llama-3.1-70b-instruct") -> CommentaryScript:
    """
    Generate a narration script from transcript segments.

    Args:
        transcript_segments: List of dicts with 'start', 'end', 'text'
        video_duration: Total video duration in seconds
        style: Commentary style ('narration', 'sports', 'summary', 'custom')
        custom_prompt: Custom instructions for style='custom'
        nvidia_api_key: NVIDIA NIM API key
        nvidia_model: Model to use

    Returns:
        CommentaryScript with timed segments
    """
    if not nvidia_api_key:
        nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")
    if not nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY required for commentary generation")

    # Build transcript context
    transcript_text = ""
    for seg in transcript_segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "")
        transcript_text += f"[{start:.1f}s - {end:.1f}s] {text}\n"

    # Build style instruction
    style_instructions = _get_style_prompt(style, custom_prompt)

    prompt = f"""You are a professional voiceover scriptwriter. Generate a narration script for a video based on its transcript.

VIDEO DURATION: {video_duration:.1f} seconds

TRANSCRIPT (with timestamps):
{transcript_text[:6000]}

STYLE: {style_instructions}

CRITICAL RULES:
1. Generate narration that covers the ENTIRE video duration
2. Split into segments that align with the transcript timestamps
3. Each segment should be 3-15 seconds of speaking time
4. Use natural pauses between segments (0.2-0.5s)
5. The narration should COMPLEMENT the video, not just repeat the transcript
6. Rephrase, summarize, add context — make it sound like a professional narrator
7. Keep total word count proportional to video length (~2.5 words per second of video)
8. For a {video_duration:.0f}s video, aim for roughly {int(video_duration * 2.5)} words total

Output ONLY valid JSON in this exact format:
{{
    "segments": [
        {{"start": 0.0, "end": 5.0, "text": "Narration text for this segment"}},
        {{"start": 5.5, "end": 12.0, "text": "Next narration segment"}},
        ...
    ]
}}

Rules for timing:
- First segment should start at 0.0 or within 1 second
- Last segment should end near {video_duration:.1f}s
- Leave small gaps (0.3-0.5s) between segments for breathing room
- Each segment's duration should be enough to speak the text at a natural pace
"""

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key,
        )

        response = client.chat.completions.create(
            model=nvidia_model,
            messages=[
                {"role": "system", "content": "You are a professional narration scriptwriter. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=4096,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        data = json.loads(raw)
        raw_segments = data.get("segments", [])

        if not raw_segments:
            raise ValueError("LLM returned empty segments")

        # Build CommentarySegment objects
        segments = []
        for i, seg in enumerate(raw_segments):
            start = float(seg.get("start", 0))
            end = float(seg.get("end", start + 5))
            text = seg.get("text", "").strip()
            if not text:
                continue

            # Clamp to video duration
            start = max(0, min(start, video_duration))
            end = max(start + 0.5, min(end, video_duration))

            # Calculate pause after (gap to next segment)
            pause = 0.3
            if i + 1 < len(raw_segments):
                next_start = float(raw_segments[i + 1].get("start", end))
                pause = max(0.2, min(next_start - end, 1.0))

            segments.append(CommentarySegment(
                start=start,
                end=end,
                text=text,
                pause_after=pause,
            ))

        full_text = " ".join(s.text for s in segments)

        logger.info(
            f"Commentary generated: {len(segments)} segments, "
            f"~{len(full_text.split())} words for {video_duration:.0f}s video"
        )

        return CommentaryScript(
            segments=segments,
            full_text=full_text,
            video_duration=video_duration,
            style=style,
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {e}")
        raise ValueError(f"AI returned invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Commentary generation failed: {e}")
        raise


def _get_style_prompt(style: str, custom_prompt: str = "") -> str:
    """Get style-specific instructions."""
    styles = {
        "narration": (
            "Professional documentary narrator style. "
            "Calm, authoritative, informative. "
            "Rephrase the content as smooth narration — don't just read the transcript. "
            "Add context, transitions, and natural flow. "
            "Think David Attenborough meets modern YouTube."
        ),
        "sports": (
            "Energetic sports commentator style. "
            "Excited, reactive, building tension. "
            "Add hype words, dramatic pauses, and reactions. "
            "Make every moment feel important."
        ),
        "summary": (
            "Concise summary recap style. "
            "Hit the key points efficiently. "
            "Skip filler, focus on main takeaways. "
            "Clear, direct, professional."
        ),
        "custom": custom_prompt or "Generate natural narration for this video.",
    }
    return styles.get(style, styles["narration"])
