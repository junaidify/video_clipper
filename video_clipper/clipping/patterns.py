"""
Pattern Knowledge Base & Matcher Module
Defines viral hook opening patterns, structural templates, and scoring logic
based on empirical short-form video engagement research.
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─── Hook Opener Regex Patterns ───
# Categorized patterns with empirical engagement weights (0.0 to 1.0)
VIRAL_HOOK_PATTERNS = {
    # ── Category 1: Contradiction / Curiosity (Highest conversion) ──
    "contradiction": {
        "weight": 1.0,
        "patterns": [
            r"\b(everything you (know|learned|thought) about .* is (wrong|a lie|backward))\b",
            r"\b(the biggest lie (you've|you have|we've|they) been told)\b",
            r"\b(stop doing .* (instead|right now|immediately))\b",
            r"\b(nobody (talks about|is talking about|knows about|tells you))\b",
            r"\b(what (they|schools?|gurus?|experts?) (don't|won't|never) tell you)\b",
            r"\b(the truth (about|behind) .* (is that|nobody))\b",
            r"\b(you('re| are) doing .* completely wrong)\b",
            r"\b(this is why you('re| are) (failing|broke|stuck|tired))\b",
            r"\b(i was wrong about)\b",
            r"\b(don't (buy|use|do|start) .* until you watch this)\b",
        ],
    },

    # ── Category 2: Secret / Insider Knowledge ──
    "secret_insider": {
        "weight": 0.95,
        "patterns": [
            r"\b(here's the (secret|formula|blueprint|framework|hack|trick))\b",
            r"\b(the (number one|#1|single most important) (thing|secret|rule|mistake))\b",
            r"\b(the (real|hidden|untold) reason (why|behind))\b",
            r"\b(how (i|we|they) actually (made|built|grew|scaled|lost))\b",
            r"\b(the (one|1) (thing|habit|skill|hack) that changed (everything|my life))\b",
            r"\b(this (one|1) (trick|tweak|change|method) (doubled|tripled|10x))\b",
            r"\b(an? (insider|industry|dark) secret)\b",
            r"\b(stealth|cheat code|unfair advantage)\b",
        ],
    },

    # ── Category 3: Listicle / Framework Hooks ──
    "listicle_framework": {
        "weight": 0.85,
        "patterns": [
            r"\b(here are (\d+|three|four|five|six|seven|top) (things|ways|signs|rules|tips|steps))\b",
            r"\b((\d+|3|4|5) (signs|reasons|mistakes|rules|lessons) you)\b",
            r"\b(step (one|1|two|2|three|3) is)\b",
            r"\b(the (\d+|3|4|5) step (framework|system|process|guide))\b",
            r"\b(if you (have|do|see) any of these (\d+|3|4|5))\b",
            r"\b(rule number (one|1|two|2|three|3))\b",
        ],
    },

    # ── Category 4: Story / Personal Revelation Hooks ──
    "story_revelation": {
        "weight": 0.90,
        "patterns": [
            r"\b(let me tell you (a|the) (story|about))\b",
            r"\b(i (lost|wasted|spent) .* (and here's|before i learned))\b",
            r"\b(when i was (\d+|young|broke|starting))\b",
            r"\b(this happened to me (when|after|while))\b",
            r"\b(the worst (mistake|day|moment) of my (life|career))\b",
            r"\b(in (\d{4}|\d+ years ago), i made a decision)\b",
            r"\b(and (then|suddenly) everything (changed|went wrong|clicked))\b",
        ],
    },

    # ── Category 5: Value Bomb / Direct Proposition ──
    "value_bomb": {
        "weight": 0.88,
        "patterns": [
            r"\b(if you('re| are) (trying to|struggling with|want to))\b",
            r"\b(how to .* in (under|less than) (\d+|60 seconds|five minutes))\b",
            r"\b(this will (save you|make you|cost you) (thousands|hours|years))\b",
            r"\b(before you (hire|buy|start|quit|invest))\b",
            r"\b(watch this if you (want|need|have))\b",
            r"\b(the fastest way to (learn|build|grow|make|get))\b",
        ],
    },

    # ── Category 6: Emotional / High Energy Reactions ──
    "emotional_peak": {
        "weight": 0.80,
        "patterns": [
            r"\b(oh my god|i can't believe|are you kidding me)\b",
            r"\b(this (blew|blows) my mind)\b",
            r"\b(i was completely (shocked|blown away|stunned))\b",
            r"\b(wait (a second|a minute|hold on|listen))\b",
            r"\b(you (won't|will not) believe what (happened|he said|she said))\b",
        ],
    },

    # ── Category 7: Hindi / Hinglish Viral Hooks ──
    "hinglish_hooks": {
        "weight": 0.92,
        "patterns": [
            r"\b(yeh (galti|secret|tarika|sach) (koi nahi|har koi))\b",
            r"\b(agar aap (bhi|yeh) (sochte|karte|chahte) ho)\b",
            r"\b(kisi ne aapko yeh nahi bataya)\b",
            r"\b(sabse badi galti jo (har|log))\b",
            r"\b(ek aisa (secret|hack|tarika|rule))\b",
            r"\b(yeh video dekhne ke baad)\b",
            r"\b(sirf (ek|1) cheez jo)\b",
            r"\b(meri baat dhyan se suno)\b",
            r"\b(kya aapko pata hai)\b",
            r"\b(asal mein (kya|sach))\b",
        ],
    },
}

# ─── Mid-Clip Transition & Value Markers ───
# Words/phrases that signal key insights inside a clip
INSIGHT_MARKERS = [
    r"\b(the key is|here's how it works|what that means is)\b",
    r"\b(the point is|at the end of the day|the bottom line)\b",
    r"\b(for example|here's an example|case in point)\b",
    r"\b(remember this|take note of this|write this down)\b",
    r"\b(and that's (why|how|when))\b",
    r"\b(result is|outcome was|difference was)\b",
]

# ─── Clip Ending / Resolution Markers ───
# Phrases that signal a good concluding moment for a clip
CONCLUSION_MARKERS = [
    r"\b(and that's the (secret|lesson|truth))\b",
    r"\b(so the next time you)\b",
    r"\b(if you do this, (you'll|you will))\b",
    r"\b(follow (for more|me for))\b",
    r"\b(let me know in the comments)\b",
    r"\b(that's how you (win|succeed|do it))\b",
]


@dataclass
class PatternMatch:
    """Represents a matched viral pattern in text."""
    category: str
    pattern: str
    matched_text: str
    weight: float
    start_char: int
    end_char: int


@dataclass
class StructuralPattern:
    """A clip arc pattern: Hook -> Development -> Climax/Insight -> Conclusion."""
    has_hook: bool = False
    has_insight: bool = False
    has_conclusion: bool = False
    hook_matches: List[PatternMatch] = field(default_factory=list)
    insight_matches: List[str] = field(default_factory=list)
    conclusion_matches: List[str] = field(default_factory=list)

    @property
    def structural_completeness(self) -> float:
        """Score 0.0-1.0 of how complete the narrative arc is."""
        score = 0.0
        if self.has_hook:
            score += 0.50  # Hook is the most critical
        if self.has_insight:
            score += 0.30  # Needs substance
        if self.has_conclusion:
            score += 0.20  # Nice ending punchline
        return score


class PatternScorer:
    """Matches transcripts against the viral patterns knowledge base."""

    def __init__(self, custom_patterns: Optional[dict] = None):
        """
        Args:
            custom_patterns: Optional dict extending or overriding VIRAL_HOOK_PATTERNS.
        """
        self.patterns = VIRAL_HOOK_PATTERNS.copy()
        if custom_patterns:
            self.patterns.update(custom_patterns)

        # Pre-compile regexes for performance
        self._compiled_hooks = {}
        for category, data in self.patterns.items():
            compiled = [
                re.compile(p, re.IGNORECASE) for p in data["patterns"]
            ]
            self._compiled_hooks[category] = (data["weight"], compiled)

        self._compiled_insights = [
            re.compile(p, re.IGNORECASE) for p in INSIGHT_MARKERS
        ]
        self._compiled_conclusions = [
            re.compile(p, re.IGNORECASE) for p in CONCLUSION_MARKERS
        ]

    def find_hook_matches(self, text: str) -> List[PatternMatch]:
        """Find all viral hook pattern matches in a text snippet."""
        matches = []
        for category, (weight, regexes) in self._compiled_hooks.items():
            for regex in regexes:
                for m in regex.finditer(text):
                    matches.append(PatternMatch(
                        category=category,
                        pattern=regex.pattern,
                        matched_text=m.group(0),
                        weight=weight,
                        start_char=m.start(),
                        end_char=m.end(),
                    ))
        return matches

    def score_hook_strength(self, text: str) -> float:
        """
        Score how strongly the text starts with or contains viral hooks (0.0 to 1.0).
        Considers both match presence and position (earlier = higher).
        """
        matches = self.find_hook_matches(text)
        if not matches:
            return 0.0

        best_score = 0.0
        text_len = max(len(text), 1)

        for m in matches:
            # Position decay: matches in the first 20% of text score higher
            pos_ratio = m.start_char / text_len
            pos_multiplier = max(0.6, 1.0 - (pos_ratio * 0.5))

            score = m.weight * pos_multiplier
            best_score = max(best_score, score)

        # Bonus for multiple hooks in same window (max +0.15)
        if len(matches) > 1:
            best_score = min(1.0, best_score + 0.05 * (len(matches) - 1))

        return round(best_score, 4)

    def analyze_structure(self, text: str) -> StructuralPattern:
        """Analyze the complete structural narrative arc of a clip candidate."""
        hook_matches = self.find_hook_matches(text)

        insight_matches = []
        for r in self._compiled_insights:
            for m in r.finditer(text):
                insight_matches.append(m.group(0))

        conclusion_matches = []
        for r in self._compiled_conclusions:
            for m in r.finditer(text):
                conclusion_matches.append(m.group(0))

        return StructuralPattern(
            has_hook=len(hook_matches) > 0,
            has_insight=len(insight_matches) > 0,
            has_conclusion=len(conclusion_matches) > 0,
            hook_matches=hook_matches,
            insight_matches=insight_matches,
            conclusion_matches=conclusion_matches,
        )
