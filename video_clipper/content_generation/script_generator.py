"""
Script Generator Module
Generates 30-60 second short-form video scripts with scene-by-scene breakdowns:
- Timestamps and visual descriptions
- On-screen text overlays
- Pexels/Pixabay visual search queries
- Dynamic transitions
"""
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional, List

from video_clipper.clipping.llm_analyzer import LLMConfig

logger = logging.getLogger(__name__)

# Last error tracker for diagnostics
_last_error = ""


@dataclass
class Scene:
    """A single scene in a short-form video script."""
    scene_number: int
    start_time: float               # seconds
    end_time: float                 # seconds
    narration: str                  # voiceover text
    visual_description: str         # what appears on screen
    visual_keywords: List[str]      # search terms for stock footage
    visual_type: str = "stock_video"  # 'stock_video', 'stock_image', 'text_card', 'ai_generated'
    text_overlay: str = ""          # on-screen text
    transition: str = "fade"        # 'fade', 'slide_left', 'dissolve', 'zoom', 'cut'

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "scene_number": self.scene_number,
            "start_time": round(self.start_time, 1),
            "end_time": round(self.end_time, 1),
            "duration": round(self.duration, 1),
            "narration": self.narration,
            "visual_description": self.visual_description,
            "visual_keywords": self.visual_keywords,
            "visual_type": self.visual_type,
            "text_overlay": self.text_overlay,
            "transition": self.transition,
        }


@dataclass
class Script:
    """A complete short-form video script."""
    title: str
    topic: str
    hook: str                       # opening sentence
    scenes: List[Scene]
    target_duration: int = 45       # target duration in seconds
    tone: str = "energetic"         # 'energetic', 'dramatic', 'informative', 'humorous', 'mysterious'
    style: str = "faceless"         # 'faceless', 'storytelling', 'listicle', 'tutorial'
    music_mood: str = "upbeat"      # 'upbeat', 'dramatic', 'chill', 'suspenseful', 'inspiring'
    hashtags: List[str] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return self.scenes[-1].end_time if self.scenes else 0.0

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "topic": self.topic,
            "hook": self.hook,
            "target_duration": self.target_duration,
            "total_duration": round(self.total_duration, 1),
            "tone": self.tone,
            "style": self.style,
            "music_mood": self.music_mood,
            "hashtags": self.hashtags,
            "scenes": [s.to_dict() for s in self.scenes],
        }


SCRIPT_SYSTEM_PROMPT = """You are a master short-form viral scriptwriter (YouTube Shorts, Instagram Reels, TikTok).
You write scripts engineered for maximum audience retention and high completion rate.

Rules for high-retention short scripts:
1. HOOK (0-3s): First sentence must instantly stop scrolling (bold statement, question, contradiction, shocking fact).
2. PACING: 130-150 words total for a 45-second video (fast, punchy sentences, zero filler).
3. STRUCTURE: 4-6 scenes. Every scene lasts 4-8 seconds with its own visual description and keywords.
4. VISUALS: Provide 2-3 specific stock footage search keywords per scene (e.g., ["person scrolling phone", "dark room", "dopamine"]).
5. RETENTION LOOP: End with an open loop or call-to-action that encourages re-watching or commenting.

Output MUST be ONLY valid JSON matching this schema (no markdown, no backticks, no explanation):
{
  "title": "Catchy YouTube Shorts Title (under 60 chars)",
  "hook": "The exact first sentence spoken in scene 1",
  "tone": "energetic",
  "style": "informative",
  "music_mood": "upbeat",
  "hashtags": ["#Shorts", "#Topic1", "#Topic2"],
  "scenes": [
    {
      "scene_number": 1,
      "start_time": 0.0,
      "end_time": 6.0,
      "narration": "Voiceover line for scene 1.",
      "visual_description": "Cinematic shot description for stock video.",
      "visual_keywords": ["keyword1", "keyword2"],
      "visual_type": "stock_video",
      "text_overlay": "PUNCHY 3-WORD TEXT",
      "transition": "fade"
    }
  ]
}"""


def generate_script(
    topic: str,
    topic_description: str = "",
    topic_keywords: Optional[List[str]] = None,
    target_duration: int = 45,
    tone: str = "energetic",
    style: str = "informative",
    llm_config: Optional[LLMConfig] = None,
) -> Optional[Script]:
    """
    Generate a complete, structured short-form video script from a topic.

    Args:
        topic: Main subject or title.
        topic_description: Extra context.
        topic_keywords: Keywords associated with topic.
        target_duration: Desired length in seconds (30-60).
        tone: 'energetic', 'dramatic', 'informative', 'humorous', 'mysterious'.
        style: 'faceless', 'storytelling', 'listicle', 'tutorial'.
        llm_config: LLMConfig with API keys.

    Returns:
        Script object or None on failure.
    """
    global _last_error
    _last_error = ""

    if llm_config is None:
        llm_config = LLMConfig.from_env()

    kw_str = ", ".join(topic_keywords) if topic_keywords else topic
    user_prompt = f"""Write a {target_duration}-second viral Shorts script about: "{topic}"

Context: {topic_description or topic}
Keywords: {kw_str}
Tone: {tone}
Style: {style}
Target duration: {target_duration} seconds (approximately {target_duration * 3} words total)

Output JSON only:"""

    logger.info(f"Generating script for: '{topic}' ({target_duration}s, tone={tone})")

    # Try providers in priority order
    raw_response = None

    if llm_config.groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=llm_config.groq_api_key)
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SCRIPT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=llm_config.groq_model,
                temperature=0.7,
                max_tokens=2048,
            )
            raw_response = resp.choices[0].message.content
            logger.info("Script generated via Groq.")
        except Exception as e:
            logger.warning(f"Groq script generation failed: {e}")
            _last_error = f"Groq error: {e}"

    if not raw_response and llm_config.gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=llm_config.gemini_api_key)
            model = genai.GenerativeModel(
                model_name=llm_config.gemini_model,
                system_instruction=SCRIPT_SYSTEM_PROMPT,
            )
            resp = model.generate_content(
                user_prompt,
                generation_config={"temperature": 0.7, "max_output_tokens": 2048},
            )
            raw_response = resp.text
            logger.info("Script generated via Gemini.")
        except Exception as e:
            logger.warning(f"Gemini script generation failed: {e}")
            _last_error = f"Gemini error: {e}"

    if not raw_response and llm_config.nvidia_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=llm_config.nvidia_api_key,
            )
            resp = client.chat.completions.create(
                model=llm_config.nvidia_model,
                messages=[
                    {"role": "system", "content": SCRIPT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=2048,
            )
            raw_response = resp.choices[0].message.content
            logger.info("Script generated via NVIDIA NIM.")
        except Exception as e:
            logger.warning(f"NVIDIA NIM script generation failed: {e}")
            _last_error = f"NVIDIA error: {e}"

    if not raw_response:
        logger.error("No LLM provider succeeded in script generation.")
        return None

    return _parse_script_json(raw_response, topic, target_duration, tone, style)


def _parse_script_json(
    raw_text: str,
    fallback_topic: str,
    target_duration: int,
    tone: str,
    style: str,
) -> Optional[Script]:
    """Parse raw LLM output into a validated Script dataclass."""
    text = raw_text.strip()

    if text.startswith("```"):
        text = re.sub(r'^```[a-z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        text = text.strip()

    start_idx = text.find("{")
    if start_idx != -1:
        depth = 0
        end_idx = -1
        in_string = False
        escape = False
        for i in range(start_idx, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
        if end_idx != -1:
            text = text[start_idx:end_idx]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse script JSON: {e}\nRaw: {text[:300]}...")
        return None

    scenes = []
    current_time = 0.0
    raw_scenes = data.get("scenes", [])

    if not raw_scenes:
        logger.error("Script JSON contained no scenes.")
        return None

    scene_dur = target_duration / len(raw_scenes)

    for i, s in enumerate(raw_scenes):
        start = float(s.get("start_time", current_time))
        end = float(s.get("end_time", current_time + scene_dur))

        if end <= start:
            start = current_time
            end = current_time + scene_dur

        current_time = end

        kws = s.get("visual_keywords", [])
        if isinstance(kws, str):
            kws = [k.strip() for k in kws.split(",")]

        scenes.append(Scene(
            scene_number=i + 1,
            start_time=start,
            end_time=end,
            narration=s.get("narration", "").strip(),
            visual_description=s.get("visual_description", "").strip(),
            visual_keywords=kws,
            visual_type=s.get("visual_type", "stock_video"),
            text_overlay=s.get("text_overlay", "").strip(),
            transition=s.get("transition", "fade"),
        ))

    return Script(
        title=data.get("title", fallback_topic)[:60],
        topic=fallback_topic,
        hook=data.get("hook", scenes[0].narration if scenes else ""),
        scenes=scenes,
        target_duration=target_duration,
        tone=tone,
        style=style,
        music_mood=data.get("music_mood", "upbeat"),
        hashtags=data.get("hashtags", ["#Shorts", "#Viral"]),
    )


def get_style_presets() -> List[dict]:
    """Return available scriptwriting style presets for UI."""
    return [
        {
            "id": "informative",
            "name": "Mind-Blowing Facts",
            "tone": "energetic",
            "style": "faceless",
            "icon": "🧠",
            "description": "Fast-paced interesting facts that hook viewers instantly.",
        },
        {
            "id": "storytelling",
            "name": "Dramatic Story",
            "tone": "dramatic",
            "style": "storytelling",
            "icon": "🎭",
            "description": "Suspenseful micro-narrative with conflict and resolution.",
        },
        {
            "id": "listicle",
            "name": "Top 3 / Top 5 List",
            "tone": "energetic",
            "style": "listicle",
            "icon": "🔢",
            "description": "Numbered listicle format with on-screen count cards.",
        },
        {
            "id": "tutorial",
            "name": "How-To / Quick Hack",
            "tone": "informative",
            "style": "tutorial",
            "icon": "⚡",
            "description": "Step-by-step actionable advice delivering immediate value.",
        },
        {
            "id": "mystery",
            "name": "Dark / Mystery",
            "tone": "mysterious",
            "style": "faceless",
            "icon": "🕵️",
            "description": "Chilling real mysteries and unexplained phenomena.",
        },
    ]
