"""
Creator Patterns Knowledge Base
================================
Distilled patterns from how top YouTube channels clip long-form into viral shorts.

Sources studied (pattern archetypes, not direct copies):
- MrBeast: challenge escalation, countdown hooks, "wait for it" tension
- Joe Rogan / podcasts: controversial take → reaction → punchline
- Lex Fridman: profound one-liner quotes, philosophical drops
- Alex Hormozi: framework reveals, "here's the thing" value bombs
- Gary Vee: rapid-fire advice, direct-to-camera energy spikes
- TED Talks: story → insight → call to action arcs
- Ali Abdaal: listicle clips, "I tested X for Y days" results
- MKBHD: tech reveals, comparison moments, verdict drops
- Huberman Lab: science claim + proof + actionable takeaway
- PowerfulJRE clips channels: heated debate moments, funny reactions

These patterns are scored and applied during transcript analysis
to weight segments that match proven viral structures.
"""
from dataclasses import dataclass, field
from typing import Optional
import re
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# PATTERN 1: HOOK OPENERS
# The first sentence that makes someone stop scrolling.
# Top channels always front-load with a hook.
# ══════════════════════════════════════════════════════════════

HOOK_OPENER_PATTERNS = [
    # Curiosity gap — makes you need to know the answer
    r"\bwhat if i told you\b",
    r"\bmost people don'?t (?:know|realize|understand)\b",
    r"\bnobody (?:talks|tells you) about\b",
    r"\bthe (?:real|actual|biggest) (?:reason|problem|secret|mistake)\b",
    r"\bhere'?s (?:the thing|what|why|how)\b",
    r"\blet me (?:tell|show|explain)\b",
    r"\bthe number one (?:mistake|thing|reason)\b",
    r"\bi'?m going to (?:show|tell|reveal|prove)\b",
    r"\byou'?re doing (?:it|this) wrong\b",
    r"\bstop doing\b",
    r"\bforget everything you (?:know|think|learned)\b",

    # Controversial / bold claim — triggers "prove it" instinct
    r"\bunpopular opinion\b",
    r"\bhot take\b",
    r"\bi don'?t care what anyone says\b",
    r"\bthis is (?:going to be |)controversial\b",
    r"\beveryone is wrong about\b",
    r"\bthe truth (?:is|about)\b",

    # Story hook — draws you into a narrative
    r"\bso (?:this|there|here'?s what) happened\b",
    r"\bi (?:was|just|recently)\b.*\band then\b",
    r"\btrue story\b",
    r"\byou won'?t believe\b",
    r"\bthis changed (?:my|everything)\b",

    # Result/transformation hook
    r"\bi (?:made|earned|lost|spent|tested|tried)\b.*\b(?:dollars?|million|thousand|percent|days?|months?|years?)\b",
    r"\bhere'?s (?:what|how) .* (?:changed|transformed|worked)\b",
    r"\bbefore and after\b",

    # Direct challenge
    r"\bif you (?:can'?t|don'?t|aren'?t)\b",
    r"\bi bet you (?:didn'?t|can'?t|don'?t)\b",
    r"\bwatch (?:this|what happens)\b",
]

# ══════════════════════════════════════════════════════════════
# PATTERN 2: VALUE BOMBS
# Dense insight delivery — the "write that down" moment.
# Hormozi, Huberman, TED style.
# ══════════════════════════════════════════════════════════════

VALUE_BOMB_PATTERNS = [
    # Framework / system reveal
    r"\bstep (?:one|two|three|1|2|3)\b",
    r"\bthe (?:framework|system|method|formula|strategy|process|secret)\b",
    r"\brule (?:number |#)?\d+\b",
    r"\bthere (?:are|is) (?:only |exactly )?\d+ (?:ways?|things?|steps?|rules?|reasons?)\b",
    r"\bthe \d+ (?:things?|mistakes?|habits?|secrets?|laws?)\b",

    # Data / proof drops
    r"\bstudies (?:show|prove|suggest|found)\b",
    r"\bresearch (?:shows?|proves?|suggests?|found)\b",
    r"\bdata (?:shows?|proves?)\b",
    r"\bscientifically (?:proven|backed)\b",
    r"\baccording to\b",
    r"\b\d+(?:\.\d+)?(?:\s)?(?:percent|%|x|times)\b",

    # Actionable takeaway
    r"\bhere'?s (?:exactly |)(?:what|how) (?:you|to)\b",
    r"\ball you (?:need|have) to do\b",
    r"\bthe (?:easiest|simplest|fastest|best) way\b",
    r"\bdo this (?:instead|every|right now)\b",
    r"\bstart (?:doing|with)\b",
    r"\bstop (?:doing|wasting)\b",
]

# ══════════════════════════════════════════════════════════════
# PATTERN 3: EMOTIONAL PEAKS
# Heated debates, laugh moments, mind-blown reactions.
# Podcast clips, reaction channels.
# ══════════════════════════════════════════════════════════════

EMOTIONAL_PEAK_PATTERNS = [
    # Disagreement / debate heat
    r"\bthat'?s (?:not true|wrong|bs|ridiculous|insane)\b",
    r"\bi (?:completely |totally |strongly )?disagree\b",
    r"\bno[,!] (?:that'?s|you'?re|it'?s)\b",
    r"\bwait[,!] what\b",
    r"\bare you (?:serious|kidding|joking)\b",
    r"\bhow (?:can|could|dare) you\b",
    r"\bthat'?s the (?:dumbest|worst|craziest)\b",

    # Mind-blown / revelation
    r"\boh my (?:god|gosh)\b",
    r"\bhold on\b",
    r"\bwait a (?:minute|second)\b",
    r"\bthink about (?:that|this|it)\b",
    r"\bthat (?:just )?blew my mind\b",
    r"\bi never (?:thought|realized|considered)\b",
    r"\bthat changes everything\b",

    # Humor / punchline
    r"\b(?:ha|haha|hahaha)\b",
    r"\bthat'?s (?:hilarious|so funny|gold)\b",

    # Inspirational peak
    r"\byou can (?:do|be|have) (?:anything|everything|it)\b",
    r"\bnever give up\b",
    r"\bbelieve in yourself\b",
    r"\bthis is (?:your|the) moment\b",
    r"\blife is too short\b",
]

# ══════════════════════════════════════════════════════════════
# PATTERN 4: STRUCTURAL SIGNALS
# Where in the video do viral clips tend to come from?
# ══════════════════════════════════════════════════════════════

@dataclass
class StructuralPattern:
    """Position-based scoring from creator analysis."""
    # Opening hook zone (first 10% of video) — most clips come from here
    opening_weight: float = 0.90
    # Topic transition zones — when the speaker pivots to a new point
    transition_boost: float = 0.25
    # Climax zone (55-75%) — the "best part" of most videos
    climax_weight: float = 0.80
    # Closing takeaway (last 12%) — "remember this" moments
    closing_weight: float = 0.65
    # Dead zones — low-value filler
    filler_penalty: float = -0.15


# ══════════════════════════════════════════════════════════════
# PATTERN 5: CLIP STRUCTURE TEMPLATES
# Successful shorts follow these narrative arcs.
# ══════════════════════════════════════════════════════════════

CLIP_ARC_TEMPLATES = {
    "hook_value_cta": {
        # Hook (1-5s) → Value delivery (10-40s) → Call to action / punchline (5-10s)
        "description": "Hook → Value → CTA — the Hormozi/Ali Abdaal format",
        "ideal_duration": (20, 45),
        "weight": 1.0,
    },
    "story_climax_lesson": {
        # Story setup (5-15s) → Build tension (10-25s) → Climax + lesson (5-15s)
        "description": "Story → Climax → Lesson — the TED/MrBeast format",
        "ideal_duration": (25, 55),
        "weight": 0.95,
    },
    "claim_proof_takeaway": {
        # Bold claim (3-8s) → Evidence/proof (10-30s) → Takeaway (5-10s)
        "description": "Claim → Proof → Takeaway — the Huberman/science format",
        "ideal_duration": (20, 50),
        "weight": 0.90,
    },
    "question_answer_reveal": {
        # Question/problem (3-10s) → Exploration (10-30s) → Answer reveal (5-10s)
        "description": "Question → Explore → Reveal — the MKBHD/tech format",
        "ideal_duration": (20, 45),
        "weight": 0.85,
    },
    "hot_take_debate": {
        # Hot take (3-8s) → Reaction/pushback (5-20s) → Resolution or cliffhanger
        "description": "Hot take → Debate → Resolution — the podcast clip format",
        "ideal_duration": (15, 40),
        "weight": 0.80,
    },
}

# ══════════════════════════════════════════════════════════════
# PATTERN 6: TRANSITION MARKERS
# Words/phrases that signal topic shifts — good clip boundaries.
# ══════════════════════════════════════════════════════════════

TRANSITION_MARKERS = [
    r"\bbut here'?s (?:the thing|what|where)\b",
    r"\bnow[,] (?:here'?s|let me|the)\b",
    r"\bso[,]? (?:anyway|moving on|next|the point is)\b",
    r"\bthe (?:second|third|next|other|biggest) (?:thing|point|reason)\b",
    r"\bon (?:top of|the other hand)\b",
    r"\bhaving said that\b",
    r"\bwhich brings me to\b",
    r"\bspeaking of\b",
    r"\blet'?s talk about\b",
    r"\bthe real question is\b",
    r"\bmore importantly\b",
]


# ══════════════════════════════════════════════════════════════
# SCORING ENGINE
# Applies all patterns to score transcript segments.
# ══════════════════════════════════════════════════════════════

class PatternScorer:
    """Scores transcript segments using creator-derived viral patterns."""

    def __init__(self):
        # Pre-compile all regex patterns
        self._hook_patterns = [re.compile(p, re.IGNORECASE) for p in HOOK_OPENER_PATTERNS]
        self._value_patterns = [re.compile(p, re.IGNORECASE) for p in VALUE_BOMB_PATTERNS]
        self._emotion_patterns = [re.compile(p, re.IGNORECASE) for p in EMOTIONAL_PEAK_PATTERNS]
        self._transition_patterns = [re.compile(p, re.IGNORECASE) for p in TRANSITION_MARKERS]
        self._structural = StructuralPattern()

    def score_segment(self, text: str, position_ratio: float,
                      is_first_segment: bool = False) -> dict:
        """
        Score a text segment against all pattern categories.

        Args:
            text: The segment text to analyze
            position_ratio: Where in the video this segment is (0.0 = start, 1.0 = end)
            is_first_segment: Whether this is the very first segment

        Returns:
            Dict with category scores and total weighted score
        """
        lower = text.lower()

        # Count pattern matches per category
        hook_hits = sum(1 for p in self._hook_patterns if p.search(lower))
        value_hits = sum(1 for p in self._value_patterns if p.search(lower))
        emotion_hits = sum(1 for p in self._emotion_patterns if p.search(lower))
        transition_hits = sum(1 for p in self._transition_patterns if p.search(lower))

        # Normalize to 0-1 (diminishing returns)
        hook_score = min(1.0, hook_hits * 0.25)
        value_score = min(1.0, value_hits * 0.20)
        emotion_score = min(1.0, emotion_hits * 0.25)
        transition_score = min(1.0, transition_hits * 0.30)

        # Structural position scoring
        position_score = self._score_position(position_ratio)

        # First segment bonus (opening hooks are gold)
        if is_first_segment:
            hook_score = min(1.0, hook_score + 0.3)

        # Weighted total
        total = (
            hook_score * 0.30 +
            value_score * 0.25 +
            emotion_score * 0.20 +
            position_score * 0.15 +
            transition_score * 0.10
        )

        return {
            "hook_opener": round(hook_score, 4),
            "value_bomb": round(value_score, 4),
            "emotional_peak": round(emotion_score, 4),
            "position": round(position_score, 4),
            "transition": round(transition_score, 4),
            "pattern_total": round(total, 4),
            "hook_hits": hook_hits,
            "value_hits": value_hits,
            "emotion_hits": emotion_hits,
        }

    def _score_position(self, ratio: float) -> float:
        """Score based on where in the video this segment falls."""
        s = self._structural

        if ratio < 0.10:
            return s.opening_weight
        elif ratio < 0.20:
            return 0.55  # post-opening, still decent
        elif 0.55 <= ratio <= 0.75:
            return s.climax_weight
        elif ratio > 0.88:
            return s.closing_weight
        else:
            return 0.35  # middle — not dead, just average

    def detect_clip_arc(self, segments_text: list) -> Optional[str]:
        """
        Try to identify which clip arc template a sequence of segments matches.
        Returns the best-matching arc name or None.
        """
        if not segments_text:
            return None

        full_text = " ".join(segments_text).lower()
        first_text = segments_text[0].lower() if segments_text else ""
        last_text = segments_text[-1].lower() if segments_text else ""

        # Check if opening has a hook
        has_hook = any(p.search(first_text) for p in self._hook_patterns)
        has_value = any(p.search(full_text) for p in self._value_patterns)
        has_emotion = any(p.search(full_text) for p in self._emotion_patterns)
        has_question = "?" in first_text

        if has_hook and has_value:
            return "hook_value_cta"
        elif has_emotion and not has_value:
            return "hot_take_debate"
        elif has_question and has_value:
            return "question_answer_reveal"
        elif has_value and not has_hook:
            return "claim_proof_takeaway"

        return None

    def is_good_clip_boundary(self, text: str) -> bool:
        """Check if a segment marks a natural clip start/end point."""
        lower = text.lower().strip()

        # Transition markers = good start boundary
        if any(p.search(lower) for p in self._transition_patterns):
            return True

        # Sentences that start with conjunctions = bad start (mid-thought)
        bad_starts = ["and ", "but ", "or ", "so ", "because ", "which ", "that "]
        if any(lower.startswith(s) for s in bad_starts):
            return False

        return True
