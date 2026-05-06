"""
LLM Analyzer Module (Optional Fallback)
Uses Groq or Gemini to analyze transcript for hooks when NLP scoring
needs a second opinion or produces weak results.

This is NOT the primary analyzer — the NLP-based analyzer.py runs first.
This module activates only when:
  1. API keys are configured in .env
  2. NLP scoring finds fewer candidates than expected
  3. User explicitly requests LLM analysis

Supports: Groq (Llama/Mixtral), Google Gemini, NVIDIA NIM
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Prompt template for hook detection
HOOK_DETECTION_PROMPT = """You are a viral content analyst. Analyze this video transcript and identify the most compelling, hook-worthy moments that would make great short-form clips (15-60 seconds) for TikTok/Reels/YouTube Shorts.

For each hook moment, identify:
1. The exact timestamp range (start_seconds - end_seconds)
2. Why this moment is compelling (quote, insight, emotional peak, key revelation, etc.)
3. A hook score from 0.0 to 1.0 (1.0 = extremely viral-worthy)

Focus on moments that:
- Deliver the essence/core message of the video
- Contain memorable quotes or statements
- Discuss something that matters most in the entire video
- Have emotional intensity or surprise
- Would make someone stop scrolling

TRANSCRIPT (with timestamps):
{transcript}

VIDEO DURATION: {duration} seconds

Respond ONLY with valid JSON array. No markdown, no explanation:
[
  {{
    "start": <float>,
    "end": <float>,
    "score": <float 0-1>,
    "hook_text": "<the key sentence(s)>",
    "reason": "<why this is hook-worthy>"
  }}
]
"""


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    # Model preferences
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.0-flash"
    nvidia_model: str = "meta/llama-3.1-70b-instruct"
    # Which provider to try first
    preferred_provider: str = "groq"  # groq, gemini, nvidia

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load config from environment variables."""
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            nvidia_model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
            preferred_provider=os.getenv("LLM_PROVIDER", "groq"),
        )

    def has_any_key(self) -> bool:
        return any([self.groq_api_key, self.gemini_api_key, self.nvidia_api_key])


def _build_transcript_text(segments: list) -> str:
    """Format transcript segments for the LLM prompt."""
    lines = []
    for seg in segments:
        ts = f"[{seg.start:.1f}s - {seg.end:.1f}s]"
        lines.append(f"{ts} {seg.text}")
    return "\n".join(lines)


def _call_groq(prompt: str, config: LLMConfig) -> str:
    """Call Groq API."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Install groq: pip install groq")

    client = Groq(api_key=config.groq_api_key)
    response = client.chat.completions.create(
        model=config.groq_model,
        messages=[
            {"role": "system", "content": "You are a viral content analyst. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _call_gemini(prompt: str, config: LLMConfig) -> str:
    """Call Google Gemini API."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Install google-generativeai: pip install google-generativeai")

    genai.configure(api_key=config.gemini_api_key)
    model = genai.GenerativeModel(config.gemini_model)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )
    return response.text


def _call_nvidia(prompt: str, config: LLMConfig) -> str:
    """Call NVIDIA NIM API (OpenAI-compatible endpoint)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=config.nvidia_api_key,
    )
    response = client.chat.completions.create(
        model=config.nvidia_model,
        messages=[
            {"role": "system", "content": "You are a viral content analyst. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def _parse_llm_response(raw_response: str) -> list:
    """Parse LLM JSON response into clip candidates."""
    # Clean up response — strip markdown code blocks if present
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Handle wrapper object like {"hooks": [...]}
    data = json.loads(text)
    if isinstance(data, dict):
        # Find the array inside
        for key, val in data.items():
            if isinstance(val, list):
                data = val
                break

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")

    candidates = []
    for item in data:
        candidates.append({
            "start": float(item.get("start", 0)),
            "end": float(item.get("end", 0)),
            "score": float(item.get("score", 0.5)),
            "hook_text": str(item.get("hook_text", "")),
            "reason": str(item.get("reason", "llm_detected")),
        })

    return candidates


def analyze_with_llm(transcript, config: Optional[LLMConfig] = None) -> list:
    """
    Analyze transcript using LLM for hook detection.

    Args:
        transcript: Transcript object from transcriber module
        config: LLM configuration (loads from .env if None)

    Returns:
        List of dicts with start, end, score, hook_text, reason
        Empty list if no LLM is available or call fails
    """
    if config is None:
        config = LLMConfig.from_env()

    if not config.has_any_key():
        logger.info("No LLM API keys configured — skipping LLM analysis")
        return []

    # Build prompt
    transcript_text = _build_transcript_text(transcript.segments)

    # Truncate if too long (most LLMs have context limits)
    if len(transcript_text) > 15000:
        logger.warning("Transcript too long for LLM, truncating to ~15000 chars")
        transcript_text = transcript_text[:15000] + "\n[...truncated...]"

    prompt = HOOK_DETECTION_PROMPT.format(
        transcript=transcript_text,
        duration=transcript.duration,
    )

    # Try providers in order
    providers = _get_provider_order(config)

    for provider_name, call_fn in providers:
        try:
            logger.info(f"Calling {provider_name} for hook analysis...")
            raw = call_fn(prompt, config)
            candidates = _parse_llm_response(raw)
            logger.info(f"{provider_name} found {len(candidates)} hook candidates")
            return candidates
        except ImportError as e:
            logger.warning(f"{provider_name} SDK not installed: {e}")
            continue
        except Exception as e:
            logger.warning(f"{provider_name} failed: {e}")
            continue

    logger.warning("All LLM providers failed — falling back to NLP-only analysis")
    return []


def _get_provider_order(config: LLMConfig) -> list:
    """Get ordered list of (name, callable) providers to try."""
    all_providers = {
        "groq": ("Groq", _call_groq, config.groq_api_key),
        "gemini": ("Gemini", _call_gemini, config.gemini_api_key),
        "nvidia": ("NVIDIA", _call_nvidia, config.nvidia_api_key),
    }

    ordered = []

    # Preferred provider first
    if config.preferred_provider in all_providers:
        name, fn, key = all_providers[config.preferred_provider]
        if key:
            ordered.append((name, fn))

    # Then the rest
    for provider_id, (name, fn, key) in all_providers.items():
        if provider_id != config.preferred_provider and key:
            ordered.append((name, fn))

    return ordered
