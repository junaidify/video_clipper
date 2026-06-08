"""
Script Generator Module
Takes a trending topic and generates an original 30-60s Short script
with hook, body, CTA. Outputs timestamped scene breakdown with visual cues.
Uses multi-provider fallback: Gemini → NVIDIA NIM → Groq.
"""

import json
import logging
import os
import re
import requests
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Scene:
    """A single scene in the script."""
    scene_number: int
    start_time: float           # seconds
    end_time: float             # seconds
    narration: str              # voiceover text for this scene
    visual_type: str            # 'stock_footage', 'motion_graphic', 'text_animation', 'stat_card'
    visual_keywords: list       # search terms for stock footage or graphic description
    visual_description: str     # detailed description of what should appear
    text_overlay: str = ""      # text to burn on screen (if any)
    transition: str = "fade"    # 'fade', 'slide_left', 'slide_right', 'dissolve', 'zoom', 'cut'
    mood: str = "neutral"       # 'energetic', 'dramatic', 'calm', 'suspenseful', 'upbeat'

    def to_dict(self) -> dict:
        return {
            "scene_number": self.scene_number,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": round(self.end_time - self.start_time, 1),
            "narration": self.narration,
            "visual_type": self.visual_type,
            "visual_keywords": self.visual_keywords,
            "visual_description": self.visual_description,
            "text_overlay": self.text_overlay,
            "transition": self.transition,
            "mood": self.mood,
        }


@dataclass
class Script:
    """Complete script for a Short."""
    title: str
    topic: str
    hook: str                   # opening hook line
    scenes: list[Scene] = field(default_factory=list)
    cta: str = ""               # call to action
    total_duration: float = 0.0
    target_duration: int = 45   # target seconds (30-60)
    tone: str = "energetic"     # overall tone
    background_music_mood: str = "upbeat"
    tags: list = field(default_factory=list)
    thumbnail_prompt: str = ""  # description for thumbnail generation

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "topic": self.topic,
            "hook": self.hook,
            "scenes": [s.to_dict() for s in self.scenes],
            "cta": self.cta,
            "total_duration": self.total_duration,
            "target_duration": self.target_duration,
            "tone": self.tone,
            "background_music_mood": self.background_music_mood,
            "tags": self.tags,
            "thumbnail_prompt": self.thumbnail_prompt,
        }


def generate_script(topic: str,
                    topic_description: str = "",
                    topic_keywords: list = None,
                    target_duration: int = 45,
                    tone: str = "energetic",
                    style: str = "informative") -> Optional[Script]:
    """
    Generate an original Short script from a trending topic.

    Args:
        topic: The trending topic title
        topic_description: Additional context about the topic
        topic_keywords: Keywords associated with the topic
        target_duration: Target video length in seconds (30-60)
        tone: 'energetic', 'dramatic', 'calm', 'humorous', 'suspenseful'
        style: 'informative', 'storytelling', 'listicle', 'reaction', 'explainer'

    Returns:
        Script object or None on failure.
        On error, sets script_generator._last_error with details.
    """
    global _last_error
    _last_error = ""

    if topic_keywords is None:
        topic_keywords = []

    num_scenes = max(3, min(8, target_duration // 7))

    prompt = _build_script_prompt(topic, topic_description, topic_keywords,
                                  target_duration, tone, style, num_scenes)

    # ── Multi-provider fallback: Gemini → NVIDIA → Groq ──
    providers = _get_available_providers()
    if not providers:
        _last_error = "No LLM API keys configured. Set GEMINI_API_KEY, NVIDIA_API_KEY, or GROQ_API_KEY in .env"
        logger.error(_last_error)
        return None

    raw = None
    used_provider = None
    errors = []

    for name, call_fn in providers:
        try:
            logger.info(f"Trying script generation with {name}...")
            raw = call_fn(prompt)
            if raw:
                used_provider = name
                break
        except Exception as e:
            err_msg = f"{name} failed: {e}"
            logger.warning(err_msg)
            errors.append(err_msg)

    if not raw:
        _last_error = "All LLM providers failed: " + " | ".join(errors)
        logger.error(_last_error)
        return None

    # ── Parse the response into a Script object ──
    try:
        return _parse_script_response(raw, topic, target_duration, tone, used_provider)
    except Exception as e:
        _last_error = f"Script parse error ({used_provider}): {e}"
        logger.error(_last_error)
        return None


# Global error tracking for pipeline to read
_last_error = ""


def _build_script_prompt(topic, description, keywords, duration, tone, style, num_scenes):
    """Build the script generation prompt."""
    return f"""You are a viral YouTube Shorts scriptwriter. Generate an ORIGINAL script about this trending topic.
Do NOT copy any existing content — create a completely fresh take.

TOPIC: {topic}
CONTEXT: {description[:500] if description else 'No additional context'}
KEYWORDS: {', '.join(keywords[:8]) if keywords else 'None'}
TARGET DURATION: {duration} seconds
TONE: {tone}
STYLE: {style}
NUMBER OF SCENES: {num_scenes}

RULES:
1. The HOOK (first 3 seconds) must be a pattern-interrupt — something that stops the scroll. Use curiosity gaps, shocking statements, or direct questions.
2. Each scene should be 5-10 seconds of narration. Keep sentences SHORT and punchy.
3. The script must feel like a conversation, not a lecture. Use "you", direct address.
4. End with a clear CTA (subscribe, comment, share).
5. For each scene, specify what VISUAL should appear — either stock footage keywords or a motion graphic description.
6. Total narration when read aloud should fit within {duration} seconds (roughly {duration * 2.5:.0f} words).
7. Make it ORIGINAL — put a unique spin, contrarian take, or unexpected angle on the topic.

Return ONLY valid JSON in this exact format:
{{
    "title": "Catchy title under 50 chars",
    "hook": "The opening hook line (first 3 seconds)",
    "cta": "The closing call to action",
    "tone": "{tone}",
    "background_music_mood": "upbeat/dramatic/chill/suspenseful/inspiring",
    "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
    "thumbnail_prompt": "Description of an eye-catching thumbnail image for this video",
    "scenes": [
        {{
            "scene_number": 1,
            "start_time": 0.0,
            "end_time": 5.0,
            "narration": "The voiceover text for this scene",
            "visual_type": "stock_footage|motion_graphic|text_animation|stat_card",
            "visual_keywords": ["keyword1", "keyword2", "keyword3"],
            "visual_description": "Detailed description of what should appear on screen",
            "text_overlay": "Short text to display on screen (or empty string)",
            "transition": "fade|slide_left|slide_right|dissolve|zoom|cut",
            "mood": "energetic|dramatic|calm|suspenseful|upbeat"
        }}
    ]
}}

IMPORTANT: Scenes must be sequential with no time gaps. Scene 1 starts at 0.0.
Total end_time of last scene should be approximately {duration} seconds.
visual_type choices:
- "stock_footage": real-world footage (people, places, nature, objects)
- "motion_graphic": animated graphics, charts, diagrams
- "text_animation": kinetic typography, words flying in
- "stat_card": a stat/fact displayed as a designed card
"""


def _extract_json_object(text: str):
    """Extract the first valid JSON object from text, handling nested braces correctly."""
    # Find first '{'
    start = text.find('{')
    if start == -1:
        return None
    # Walk through counting braces to find matching '}'
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\' and in_string:
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                # Return a match-like object with .group()
                class _Match:
                    def __init__(self, s): self._s = s
                    def group(self): return self._s
                return _Match(text[start:i+1])
    return None


def _parse_script_response(raw: str, topic: str, target_duration: int,
                           tone: str, provider: str) -> Optional[Script]:
    """Parse LLM response text into a Script object."""
    json_match = _extract_json_object(raw)
    if not json_match:
        raise ValueError(f"No JSON found in {provider} response")

    data = json.loads(json_match.group())

    scenes = []
    for s in data.get("scenes", []):
        scene = Scene(
            scene_number=s.get("scene_number", 0),
            start_time=float(s.get("start_time", 0)),
            end_time=float(s.get("end_time", 0)),
            narration=s.get("narration", ""),
            visual_type=s.get("visual_type", "stock_footage"),
            visual_keywords=s.get("visual_keywords", []),
            visual_description=s.get("visual_description", ""),
            text_overlay=s.get("text_overlay", ""),
            transition=s.get("transition", "fade"),
            mood=s.get("mood", "neutral"),
        )
        scenes.append(scene)

    if not scenes:
        raise ValueError("No scenes found in response")

    total_dur = scenes[-1].end_time if scenes else 0

    script = Script(
        title=data.get("title", topic[:50]),
        topic=topic,
        hook=data.get("hook", ""),
        scenes=scenes,
        cta=data.get("cta", ""),
        total_duration=total_dur,
        target_duration=target_duration,
        tone=data.get("tone", tone),
        background_music_mood=data.get("background_music_mood", "upbeat"),
        tags=data.get("tags", []),
        thumbnail_prompt=data.get("thumbnail_prompt", ""),
    )

    logger.info(
        f"Script generated via {provider}: '{script.title}' — "
        f"{len(scenes)} scenes, {total_dur:.0f}s"
    )
    return script


# ── LLM Provider Implementations ──

def _get_available_providers() -> list[tuple]:
    """Return list of (name, call_fn) for providers with valid API keys."""
    providers = []

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        providers.append(("Gemini", lambda p: _call_gemini(p, gemini_key)))

    nvidia_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if nvidia_key:
        providers.append(("NVIDIA", lambda p: _call_nvidia(p, nvidia_key)))

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        providers.append(("Groq", lambda p: _call_groq(p, groq_key)))

    return providers


def _call_gemini(prompt: str, api_key: str) -> str:
    """Call Google Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    # Try models in order
    models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    last_err = None
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response.text:
                return response.text.strip()
        except Exception as e:
            last_err = f"{model_name}: {e}"
            logger.warning(f"Gemini model {model_name} failed: {e}")
    raise RuntimeError(f"All Gemini models failed. Last: {last_err}")


def _call_nvidia(prompt: str, api_key: str) -> str:
    """Call NVIDIA NIM API (OpenAI-compatible)."""
    # Try models in order: newest first, then fallbacks
    models = [
        "meta/llama-3.3-70b-instruct",
        "meta/llama-3.1-70b-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct",
    ]
    last_err = None
    for model in models:
        try:
            resp = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                    "max_tokens": 4096,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            last_err = f"{model}: HTTP {resp.status_code}"
            logger.warning(f"NVIDIA model {model} failed: {resp.status_code}")
        except Exception as e:
            last_err = f"{model}: {e}"
            logger.warning(f"NVIDIA model {model} error: {e}")
    raise RuntimeError(f"All NVIDIA models failed. Last: {last_err}")


def _call_groq(prompt: str, api_key: str) -> str:
    """Call Groq API (OpenAI-compatible)."""
    # llama-3.1-70b-versatile was deprecated; try current models
    models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ]
    last_err = None
    for model in models:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                    "max_tokens": 4096,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            last_err = f"{model}: HTTP {resp.status_code}"
            logger.warning(f"Groq model {model} failed: {resp.status_code}")
        except Exception as e:
            last_err = f"{model}: {e}"
            logger.warning(f"Groq model {model} error: {e}")
    raise RuntimeError(f"All Groq models failed. Last: {last_err}")


def regenerate_scene(script: Script, scene_number: int,
                     feedback: str = "") -> Optional[Scene]:
    """
    Regenerate a single scene with optional user feedback.
    Useful for iterating on specific parts of the script.
    """
    old_scene = None
    for s in script.scenes:
        if s.scene_number == scene_number:
            old_scene = s
            break

    if not old_scene:
        return None

    prompt = f"""Rewrite this single scene from a YouTube Short script.

FULL SCRIPT TOPIC: {script.topic}
SCRIPT TONE: {script.tone}

CURRENT SCENE #{scene_number}:
- Narration: "{old_scene.narration}"
- Visual: {old_scene.visual_description}
- Time: {old_scene.start_time}s - {old_scene.end_time}s

{f'USER FEEDBACK: {feedback}' if feedback else 'Make it more engaging and punchy.'}

Keep the same time window ({old_scene.start_time}s - {old_scene.end_time}s).
Return ONLY valid JSON for the single scene in the same format:
{{
    "scene_number": {scene_number},
    "start_time": {old_scene.start_time},
    "end_time": {old_scene.end_time},
    "narration": "improved narration",
    "visual_type": "stock_footage|motion_graphic|text_animation|stat_card",
    "visual_keywords": ["keyword1", "keyword2"],
    "visual_description": "what should appear",
    "text_overlay": "",
    "transition": "{old_scene.transition}",
    "mood": "{old_scene.mood}"
}}"""

    # Use fallback chain
    providers = _get_available_providers()
    if not providers:
        return None

    for name, call_fn in providers:
        try:
            raw = call_fn(prompt)
            if not raw:
                continue
            json_match = _extract_json_object(raw)
            if not json_match:
                continue
            s = json.loads(json_match.group())
            return Scene(
                scene_number=s.get("scene_number", scene_number),
                start_time=float(s.get("start_time", old_scene.start_time)),
                end_time=float(s.get("end_time", old_scene.end_time)),
                narration=s.get("narration", old_scene.narration),
                visual_type=s.get("visual_type", old_scene.visual_type),
                visual_keywords=s.get("visual_keywords", old_scene.visual_keywords),
                visual_description=s.get("visual_description", old_scene.visual_description),
                text_overlay=s.get("text_overlay", ""),
                transition=s.get("transition", old_scene.transition),
                mood=s.get("mood", old_scene.mood),
            )
        except Exception as e:
            logger.warning(f"Scene regen via {name} failed: {e}")

    return None


def get_style_presets() -> list[dict]:
    """Return available script style presets for UI."""
    return [
        {
            "id": "informative",
            "label": "Informative",
            "description": "Facts and insights delivered fast",
            "tone": "energetic",
            "icon": "📊",
        },
        {
            "id": "storytelling",
            "label": "Storytelling",
            "description": "Narrative arc with suspense and payoff",
            "tone": "dramatic",
            "icon": "📖",
        },
        {
            "id": "listicle",
            "label": "Listicle",
            "description": "Top 3/5 format — ranked items",
            "tone": "upbeat",
            "icon": "📋",
        },
        {
            "id": "reaction",
            "label": "Hot Take",
            "description": "Contrarian opinion on trending topic",
            "tone": "energetic",
            "icon": "🔥",
        },
        {
            "id": "explainer",
            "label": "Explainer",
            "description": "Break down complex topic simply",
            "tone": "calm",
            "icon": "💡",
        },
    ]
