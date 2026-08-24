"""
Commentary Script Generator Module
Generates an AI voiceover commentary track that reacts to, explains, or adds context
to an existing video. Analyzes transcript segments and produces timed commentary snippets.
"""
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional, List

from video_clipper.clipping.transcriber import Transcript
from video_clipper.clipping.llm_analyzer import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class CommentarySegment:
    """A single piece of commentary timed to a video moment."""
    start: float                    # target video start time (seconds)
    end: float                      # target video end time (seconds)
    text: str                       # what the AI voice says
    pause_after: float = 0.3        # seconds of silence after this line
    emotion: str = "neutral"        # 'excited', 'curious', 'shocked', 'analytical'

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 1),
            "end": round(self.end, 1),
            "duration": round(self.duration, 1),
            "text": self.text,
            "pause_after": self.pause_after,
            "emotion": self.emotion,
        }


@dataclass
class CommentaryScript:
    """Full commentary script for a video."""
    video_title: str
    video_duration: float
    style: str                      # 'reaction', 'educational', 'humorous', 'sports'
    language: str = "en"
    segments: List[CommentarySegment] = field(default_factory=list)
    total_speech_duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "video_title": self.video_title,
            "video_duration": round(self.video_duration, 1),
            "style": self.style,
            "language": self.language,
            "total_speech_duration": round(self.total_speech_duration, 1),
            "segments": [s.to_dict() for s in self.segments],
        }


COMMENTARY_PROMPTS = {
    "reaction": """You are an enthusiastic, charismatic YouTube reactor giving dynamic commentary on a video clip.
React with genuine emotion, humor, and surprise. Keep it punchy and engaging.""",

    "educational": """You are an insightful educator providing high-value context, background facts, and explanations
over a video clip. Deliver clean, authoritative insights that make the viewer smarter.""",

    "humorous": """You are a witty, satirical commentator roasting and joking about the video clip.
Keep it light, humorous, and entertaining without being offensive.""",

    "sports": """You are an energetic play-by-play sports caster delivering high-octane commentary
and breakdown on the action unfolding in the video.""",
}


def generate_commentary(
    transcript: Transcript,
    video_title: str = "",
    style: str = "reaction",
    target_coverage: float = 0.6,
    llm_config: Optional[LLMConfig] = None,
) -> Optional[CommentaryScript]:
    """
    Generate timed commentary lines matching the transcript timeline.

    Args:
        transcript: Transcript of the source video.
        video_title: Optional video title for context.
        style: 'reaction', 'educational', 'humorous', 'sports'.
        target_coverage: Fraction of total duration covered by speech (0.3 - 0.8).
        llm_config: Optional LLMConfig.

    Returns:
        CommentaryScript or None on failure.
    """
    if llm_config is None:
        llm_config = LLMConfig.from_env()

    if not llm_config.has_any_key():
        logger.warning("No LLM API keys configured for commentary generation.")
        return None

    # Format transcript snippets
    lines = []
    for s in transcript.segments:
        lines.append(f"[{s.start:.1f}s - {s.end:.1f}s] {s.text}")
    transcript_summary = "\n".join(lines[:30])

    system_prompt = COMMENTARY_PROMPTS.get(style, COMMENTARY_PROMPTS["reaction"])

    user_prompt = f"""VIDEO TITLE: "{video_title or 'Clip'}"
TOTAL DURATION: {transcript.duration:.1f} seconds

TRANSCRIPT TIMELINE:
{transcript_summary}

TASK:
Write a commentary track for this video.
Produce 3 to 6 well-timed commentary lines that react to or explain specific moments.
Ensure start and end timestamps fall strictly within 0.0 and {transcript.duration:.1f} seconds.

Output ONLY valid JSON:
{{
  "segments": [
    {{
      "start": 2.0,
      "end": 8.0,
      "text": "Notice how they set up this moment right here...",
      "emotion": "curious"
    }}
  ]
}}"""

    raw_response = None
    if llm_config.groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=llm_config.groq_api_key)
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=llm_config.groq_model,
                temperature=0.6,
                max_tokens=1500,
            )
            raw_response = resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq commentary generation failed: {e}")

    if not raw_response and llm_config.gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=llm_config.gemini_api_key)
            model = genai.GenerativeModel(
                model_name=llm_config.gemini_model,
                system_instruction=system_prompt,
            )
            resp = model.generate_content(user_prompt)
            raw_response = resp.text
        except Exception as e:
            logger.warning(f"Gemini commentary generation failed: {e}")

    if not raw_response:
        return None

    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r'^```[a-z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        text = text.strip()

    try:
        data = json.loads(text)
        segments = []
        for s in data.get("segments", []):
            start = float(s.get("start", 0.0))
            end = float(s.get("end", start + 5.0))
            segments.append(CommentarySegment(
                start=start,
                end=end,
                text=s.get("text", "").strip(),
                emotion=s.get("emotion", "neutral"),
            ))

        total_speech = sum(s.duration for s in segments)
        return CommentaryScript(
            video_title=video_title,
            video_duration=transcript.duration,
            style=style,
            segments=segments,
            total_speech_duration=total_speech,
        )
    except Exception as e:
        logger.error(f"Failed to parse commentary JSON: {e}")
        return None
