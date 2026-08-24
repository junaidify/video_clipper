"""
LLM Hook Analyzer Module
Optional fallback/enhancement layer using multi-provider LLM APIs:
- Groq (fast, free tier: llama-3.3-70b, mixtral-8x7b)
- Google Gemini (gemini-2.0-flash, gemini-1.5-pro)
- NVIDIA NIM (free trial: meta/llama-3.1-70b-instruct)
- OpenAI-compatible endpoints
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM-based hook detection."""
    # Provider keys (read from env or passed directly)
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Model preferences per provider
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.0-flash"
    nvidia_model: str = "meta/llama-3.1-70b-instruct"
    openai_model: str = "gpt-4o-mini"

    # Preferred provider order
    provider_priority: List[str] = None

    def __post_init__(self):
        if self.provider_priority is None:
            self.provider_priority = ["groq", "gemini", "nvidia", "openai"]

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load API keys from environment variables."""
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    @property
    def preferred_provider(self) -> str:
        for p in (self.provider_priority or ["groq", "gemini", "nvidia", "openai"]):
            if getattr(self, f"{p}_api_key", None):
                return p
        return "none"

    def has_any_key(self) -> bool:
        """Check if at least one LLM provider is configured."""
        return bool(
            self.groq_api_key or self.gemini_api_key or
            self.nvidia_api_key or self.openai_api_key
        )


HOOK_DETECTION_SYSTEM_PROMPT = """You are an expert short-form video editor and viral content strategist.
Your job is to analyze video transcripts and identify the absolute BEST moments to clip into 15-60 second standalone clips for TikTok, Instagram Reels, and YouTube Shorts.

For each clip candidate, you must identify:
1. Exact start and end timestamps (in seconds)
2. The hook opener sentence (the first 3-5 seconds that stops the scroll)
3. Why this moment will perform well (psychological trigger, emotional peak, secret, value bomb, controversy)
4. A virality score from 0.0 to 1.0

Rules for clips:
- Duration MUST be between 15 and 60 seconds
- The clip MUST start with a strong hook (question, bold claim, story opener, contradiction)
- The clip MUST deliver on the hook's promise (complete thought / insight)
- Clips should NOT cut off mid-sentence
- Prioritize: high emotion, surprising facts, counter-intuitive advice, actionable frameworks, memorable stories

Output format: Return ONLY a valid JSON array of objects, with no surrounding text or markdown formatting.
Each object must have these exact keys:
[
  {
    "start": 12.5,
    "end": 45.0,
    "hook_text": "Here's the secret nobody tells you about...",
    "reason": "Contradiction hook followed by actionable 3-step framework",
    "score": 0.92
  }
]"""


def _call_groq(transcript_text: str, config: LLMConfig, max_clips: int) -> Optional[str]:
    """Call Groq API for hook detection."""
    try:
        from groq import Groq
        client = Groq(api_key=config.groq_api_key)

        prompt = f"Analyze this transcript and find up to {max_clips} viral clip candidates:\n\n{transcript_text}"

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": HOOK_DETECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=config.groq_model,
            temperature=0.3,
            max_tokens=2048,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq API call failed: {e}")
        return None


def _call_gemini(transcript_text: str, config: LLMConfig, max_clips: int) -> Optional[str]:
    """Call Google Gemini API for hook detection."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=config.gemini_api_key)

        model = genai.GenerativeModel(
            model_name=config.gemini_model,
            system_instruction=HOOK_DETECTION_SYSTEM_PROMPT,
        )

        prompt = f"Analyze this transcript and find up to {max_clips} viral clip candidates:\n\n{transcript_text}"
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 2048}
        )
        return response.text
    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}")
        return None


def _call_nvidia(transcript_text: str, config: LLMConfig, max_clips: int) -> Optional[str]:
    """Call NVIDIA NIM API (OpenAI-compatible) for hook detection."""
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=config.nvidia_api_key,
        )

        prompt = f"Analyze this transcript and find up to {max_clips} viral clip candidates:\n\n{transcript_text}"

        completion = client.chat.completions.create(
            model=config.nvidia_model,
            messages=[
                {"role": "system", "content": HOOK_DETECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.warning(f"NVIDIA NIM API call failed: {e}")
        return None


def _call_openai(transcript_text: str, config: LLMConfig, max_clips: int) -> Optional[str]:
    """Call standard OpenAI API for hook detection."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.openai_api_key)

        prompt = f"Analyze this transcript and find up to {max_clips} viral clip candidates:\n\n{transcript_text}"

        completion = client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {"role": "system", "content": HOOK_DETECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.warning(f"OpenAI API call failed: {e}")
        return None


# Dispatch table
_PROVIDERS = {
    "groq": (_call_groq, lambda c: c.groq_api_key),
    "gemini": (_call_gemini, lambda c: c.gemini_api_key),
    "nvidia": (_call_nvidia, lambda c: c.nvidia_api_key),
    "openai": (_call_openai, lambda c: c.openai_api_key),
}


def _parse_llm_response(raw_response: str) -> List[dict]:
    """Clean and parse LLM JSON response into list of dicts."""
    text = raw_response.strip()

    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            # Validate each candidate
            valid = []
            for item in data:
                if (
                    isinstance(item, dict) and
                    "start" in item and "end" in item and
                    isinstance(item["start"], (int, float)) and
                    isinstance(item["end"], (int, float)) and
                    item["end"] > item["start"]
                ):
                    valid.append({
                        "start": float(item["start"]),
                        "end": float(item["end"]),
                        "hook_text": str(item.get("hook_text", "")),
                        "reason": str(item.get("reason", "LLM-identified viral moment")),
                        "score": float(item.get("score", 0.7)),
                    })
            return valid
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {e}\nRaw text: {text[:200]}...")

    return []


def analyze_with_llm(
    transcript,
    config: Optional[LLMConfig] = None,
    max_clips: int = 10,
) -> List[dict]:
    """
    Run LLM-powered hook analysis on a transcript.
    Tries providers in configured priority order with automatic fallback.

    Args:
        transcript: Transcript object with segments.
        config: LLMConfig with API keys (loads from env if None).
        max_clips: Maximum candidates to request.

    Returns:
        List of dicts: [{"start": float, "end": float, "hook_text": str, "reason": str, "score": float}]
    """
    if config is None:
        config = LLMConfig.from_env()

    if not config.has_any_key():
        logger.info("No LLM API keys configured — skipping LLM hook analysis.")
        return []

    # Format transcript with timestamps for LLM input
    lines = []
    for s in transcript.segments:
        lines.append(f"[{s.start:.1f}s -> {s.end:.1f}s] {s.text}")
    transcript_text = "\n".join(lines)

    # Try each configured provider in priority order
    for provider_name in config.provider_priority:
        if provider_name not in _PROVIDERS:
            continue

        caller_fn, key_getter = _PROVIDERS[provider_name]
        if not key_getter(config):
            continue

        logger.info(f"Attempting hook analysis with LLM provider: '{provider_name}'...")
        raw_result = caller_fn(transcript_text, config, max_clips)

        if raw_result:
            candidates = _parse_llm_response(raw_result)
            if candidates:
                logger.info(
                    f"LLM provider '{provider_name}' returned {len(candidates)} valid clip candidates."
                )
                return candidates

        logger.warning(f"Provider '{provider_name}' returned no valid candidates, trying fallback...")

    logger.warning("All configured LLM providers failed or returned no candidates.")
    return []
