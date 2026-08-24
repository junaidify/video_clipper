"""
Content Analyzer Module
Multi-factor heuristic scoring engine that analyzes Whisper transcripts to find
the most engaging, viral moments in a video for short-form clipping.

Combines:
1. TF-IDF uniqueness scoring
2. Quote / Dialogue density detection
3. Viral hook keyword scoring
4. Sentiment and emotional intensity scoring
5. Video position heuristic
6. Empirical regex viral pattern scoring (patterns.py)
7. Optional LLM-assisted hook detection fallback (llm_analyzer.py)
8. Optional learned creator style profile weighting (trainer.py)
"""
import math
import re
import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

from video_clipper.config import AnalyzerConfig
from video_clipper.clipping.transcriber import Transcript, TranscriptSegment
from video_clipper.clipping.patterns import PatternScorer
from video_clipper.clipping.llm_analyzer import LLMConfig, analyze_with_llm

logger = logging.getLogger(__name__)


# Common English stopwords to ignore during TF-IDF calculation
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further had hadn't has
hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought
our ours ourselves out over own same shan't she she'd she'll she's should shouldn't
so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's with won't would wouldn't you you'd
you'll you're you've your yours yourself yourselves yeah like um uh oh okay ok well
just really know mean right going gonna wanna got get let say said things thing
""".split())


@dataclass
class ClipCandidate:
    """A scored candidate segment for clipping."""
    start: float                  # start timestamp in seconds
    end: float                    # end timestamp in seconds
    score: float                  # composite score 0.0 - 1.0
    hook_text: str                # the opening sentence/hook
    reason: str                   # explanation of why this was picked
    segment_ids: List[int] = field(default_factory=list)
    scores_breakdown: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


class ContentAnalyzer:
    """Analyzes a video transcript to identify viral clip moments."""

    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        use_llm: bool = False,
        creator_profile: Optional[dict] = None,
    ):
        """
        Args:
            config: AnalyzerConfig instance.
            use_llm: Whether to attempt LLM-assisted hook analysis.
            creator_profile: Optional training profile dict from PatternTrainer.
        """
        self.config = config or AnalyzerConfig()
        self.use_llm = use_llm
        self.creator_profile = creator_profile
        self.pattern_scorer = PatternScorer()

    def analyze(self, transcript: Transcript) -> List[ClipCandidate]:
        """
        Analyze the transcript and return top-ranked clip candidates.

        Args:
            transcript: Transcript object with segments.

        Returns:
            List of ClipCandidate objects sorted by start time.
        """
        if not transcript.segments:
            logger.warning("Empty transcript provided to analyzer.")
            return []

        logger.info(f"Analyzing transcript ({len(transcript.segments)} segments, {transcript.duration:.1f}s)...")

        # 1. Try LLM analysis first if enabled
        llm_candidates = []
        if self.use_llm:
            try:
                llm_results = analyze_with_llm(transcript, max_clips=self.config.max_clips)
                for item in llm_results:
                    llm_candidates.append(ClipCandidate(
                        start=item["start"],
                        end=item["end"],
                        score=item["score"],
                        hook_text=item.get("hook_text", ""),
                        reason=f"[LLM] {item.get('reason', 'AI-detected hook')}",
                        scores_breakdown={"llm_score": item["score"]},
                    ))
            except Exception as e:
                logger.warning(f"LLM hook analysis encountered an error, falling back to heuristics: {e}")

        # 2. Build sliding windows from transcript segments
        windows = self._build_sliding_windows(transcript)

        # 3. Compute document-level TF-IDF across all segments
        tfidf_scores = self._compute_tfidf(transcript.segments)

        # 4. Score each sliding window
        heuristic_candidates = []
        for w in windows:
            score, breakdown, reason = self._score_window(w, tfidf_scores, transcript.duration)

            if score >= self.config.min_hook_score:
                hook_text = w["segments"][0].text if w["segments"] else ""
                heuristic_candidates.append(ClipCandidate(
                    start=w["start"],
                    end=w["end"],
                    score=round(score, 4),
                    hook_text=hook_text,
                    reason=reason,
                    segment_ids=[s.id for s in w["segments"]],
                    scores_breakdown=breakdown,
                ))

        # 5. Merge heuristic candidates and LLM candidates
        all_candidates = llm_candidates + heuristic_candidates

        # 6. Deduplicate & merge overlapping candidates
        merged = self._merge_overlapping(all_candidates)

        # 7. Sort by score descending and take top N
        merged.sort(key=lambda c: c.score, reverse=True)
        top_candidates = merged[:self.config.max_clips]

        # 8. Sort final selection chronologically
        top_candidates.sort(key=lambda c: c.start)

        logger.info(f"Analysis complete: {len(top_candidates)} clips selected from {len(all_candidates)} candidates.")
        for i, c in enumerate(top_candidates):
            logger.info(f"  Clip {i+1}: [{c.start:.1f}s -> {c.end:.1f}s] (score={c.score:.3f}, reason='{c.reason}')")

        return top_candidates

    def _build_sliding_windows(self, transcript: Transcript) -> List[dict]:
        """
        Group adjacent transcript segments into time-based sliding windows.
        Windows are built around sentence boundaries.
        """
        windows = []
        segments = transcript.segments
        n = len(segments)

        # Try various window durations around the target window_size (e.g. 20s, 30s, 45s, 60s)
        window_targets = [20.0, 30.0, 45.0, 60.0]

        for target_dur in window_targets:
            for start_idx in range(n):
                window_segs = []
                current_dur = 0.0

                for end_idx in range(start_idx, n):
                    seg = segments[end_idx]
                    window_segs.append(seg)
                    current_dur = window_segs[-1].end - window_segs[0].start

                    if current_dur >= target_dur or end_idx == n - 1:
                        # Only keep if within acceptable short-form bounds (15-65s)
                        if 15.0 <= current_dur <= 65.0:
                            combined_text = " ".join(s.text for s in window_segs)
                            windows.append({
                                "start": window_segs[0].start,
                                "end": window_segs[-1].end,
                                "duration": current_dur,
                                "text": combined_text,
                                "segments": window_segs,
                            })
                        break

        # Deduplicate identical (start, end) pairs
        seen = set()
        unique_windows = []
        for w in windows:
            key = (round(w["start"], 1), round(w["end"], 1))
            if key not in seen:
                seen.add(key)
                unique_windows.append(w)

        return unique_windows

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words, stripping punctuation and stopwords."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return [w for w in words if w not in STOPWORDS]

    def _compute_tfidf(self, segments: List[TranscriptSegment]) -> dict:
        """Compute TF-IDF score for each segment relative to the full transcript corpus."""
        total_docs = max(len(segments), 1)
        doc_tokens = [self._tokenize(s.text) for s in segments]

        # Calculate document frequency (DF)
        df = Counter()
        for tokens in doc_tokens:
            for term in set(tokens):
                df[term] += 1

        # Calculate TF-IDF per segment
        scores = {}
        max_score = 0.001
        for i, (seg, tokens) in enumerate(zip(segments, doc_tokens)):
            if not tokens:
                scores[seg.id] = 0.0
                continue

            tf = Counter(tokens)
            tfidf_sum = 0.0
            for term, count in tf.items():
                tf_val = count / len(tokens)
                idf_val = math.log((total_docs + 1) / (df[term] + 1)) + 1
                tfidf_sum += tf_val * idf_val

            scores[seg.id] = tfidf_sum
            max_score = max(max_score, tfidf_sum)

        # Normalize 0.0 - 1.0
        return {seg_id: val / max_score for seg_id, val in scores.items()}

    def _score_window(
        self,
        window: dict,
        tfidf_scores: dict,
        total_duration: float,
    ) -> Tuple[float, dict, str]:
        """
        Score a single sliding window across all dimensions.
        Returns: (composite_score, score_breakdown, primary_reason)
        """
        text = window["text"]
        segments = window["segments"]
        opener_text = segments[0].text if segments else text

        # 1. TF-IDF Information Density
        seg_tfidf = [tfidf_scores.get(s.id, 0.0) for s in segments]
        tfidf_score = sum(seg_tfidf) / len(seg_tfidf) if seg_tfidf else 0.0

        # 2. Quote / First-Person Storytelling Density
        quote_score = self._score_quotes(text)

        # 3. Viral Hook Keyword Presence
        keyword_score = self._score_keywords(opener_text)

        # 4. Sentiment & Emotional Intensity
        sentiment_score = self._score_sentiment(text)

        # 5. Position Heuristic (Hook zones: 0-15% and 65-85% climax)
        position_score = self._score_position(window["start"], total_duration)

        # 6. Empirical Viral Regex Pattern Matching
        pattern_score = self.pattern_scorer.score_hook_strength(opener_text)
        structural_arc = self.pattern_scorer.analyze_structure(text)
        structure_bonus = structural_arc.structural_completeness * 0.15

        # 7. Creator Profile Personalization (if trained profile exists)
        profile_bonus = 0.0
        if self.creator_profile:
            profile_bonus = self._score_against_profile(window)

        # Compute weighted sum
        cfg = self.config
        composite = (
            cfg.tfidf_weight * tfidf_score +
            cfg.quote_weight * quote_score +
            cfg.keyword_weight * keyword_score +
            cfg.sentiment_weight * sentiment_score +
            cfg.position_weight * position_score +
            0.20 * pattern_score +
            structure_bonus +
            profile_bonus
        )

        # Normalize / clamp to 0.0 - 1.0
        final_score = min(1.0, max(0.0, composite))

        breakdown = {
            "tfidf": round(tfidf_score, 3),
            "quote": round(quote_score, 3),
            "keyword": round(keyword_score, 3),
            "sentiment": round(sentiment_score, 3),
            "position": round(position_score, 3),
            "pattern": round(pattern_score, 3),
            "structure": round(structure_bonus, 3),
        }

        # Identify primary reason for high score
        reasons = {
            "viral_pattern_match": pattern_score,
            "hook_keywords_detected": keyword_score,
            "emotionally_charged": sentiment_score,
            "high_information_density": tfidf_score,
            "engaging_storytelling": quote_score,
        }
        primary_reason = max(reasons, key=reasons.get)

        return final_score, breakdown, primary_reason

    def _score_quotes(self, text: str) -> float:
        """Score based on dialogue, first-person storytelling, and conversational engagement."""
        score = 0.0
        lower = text.lower()

        # Direct quotes or speech marks
        quote_count = text.count('"') + text.count("'")
        score += min(0.3, quote_count * 0.05)

        # First-person storytelling words ("I realized", "he told me", "we decided")
        story_cues = [
            r"\bi (was|realized|decided|thought|learned|saw|felt|spent|lost|made)\b",
            r"\b(he|she|they) (said|told me|asked|screamed|whispered)\b",
            r"\bmy (friend|mom|dad|mentor|boss|team|life)\b",
        ]
        for cue in story_cues:
            if re.search(cue, lower):
                score += 0.15

        # Questions (engagement hooks)
        score += min(0.2, text.count("?") * 0.1)

        return min(1.0, score)

    def _score_keywords(self, text: str) -> float:
        """Score presence of configured viral hook keywords."""
        lower = text.lower()
        matched = 0
        for kw in self.config.hook_keywords:
            if kw.lower() in lower:
                matched += 1

        # Diminishing returns after 3 keywords
        if matched == 0:
            return 0.0
        elif matched == 1:
            return 0.5
        elif matched == 2:
            return 0.8
        else:
            return 1.0

    def _score_sentiment(self, text: str) -> float:
        """Heuristic emotional intensity scoring."""
        score = 0.0

        # Exclamations
        score += min(0.25, text.count("!") * 0.08)

        # High-intensity / dramatic words
        intensity_words = [
            "insane", "crazy", "unbelievable", "shocking", "mind-blowing",
            "terrible", "worst", "best", "greatest", "biggest", "massive",
            "destroy", "fail", "ruin", "secret", "never", "always",
            "impossible", "genius", "danger", "warning", "deadly",
            "pyaar", "dhamaka", "zabardast", "khatarnaak", "bawaal",
        ]
        lower = text.lower()
        for w in intensity_words:
            if w in lower:
                score += 0.10

        # Contrast words (turning points in a story)
        contrast_words = ["but then", "however", "the truth is", "in reality", "suddenly", "all of a sudden"]
        for cw in contrast_words:
            if cw in lower:
                score += 0.15

        return min(1.0, score)

    def _score_position(self, start_time: float, total_duration: float) -> float:
        """Position heuristic: hooks in the beginning and climaxes near 70-80% of video."""
        if total_duration <= 0:
            return 0.5

        ratio = start_time / total_duration

        if ratio <= 0.15:
            # Opening hook zone: highest priority
            return 0.9 - (ratio * 0.5)
        elif 0.65 <= ratio <= 0.85:
            # Climax / payoff zone: high priority
            return 0.70
        elif ratio > 0.85:
            # Outro zone: lower priority
            return 0.40
        else:
            # Middle zone: baseline
            return 0.30

    def _score_against_profile(self, window: dict) -> float:
        """Score candidate against a user's learned training profile."""
        if not self.creator_profile:
            return 0.0

        bonus = 0.0
        text = window["text"].lower()

        # Check opening keyword matches from profile
        opening_keywords = self.creator_profile.get("opening_keywords", [])
        opener = window["segments"][0].text.lower() if window.get("segments") else text
        for kw in opening_keywords[:10]:
            if kw.lower() in opener:
                bonus += 0.04

        # Check content keyword matches
        content_keywords = self.creator_profile.get("content_keywords", [])
        for kw in content_keywords[:15]:
            if kw.lower() in text:
                bonus += 0.02

        # Duration match bonus
        avg_dur = self.creator_profile.get("avg_clip_duration", 0)
        if avg_dur > 0:
            dur_diff = abs(window["duration"] - avg_dur)
            if dur_diff < 5.0:
                bonus += 0.05

        return min(0.20, bonus)

    def _merge_overlapping(self, candidates: List[ClipCandidate]) -> List[ClipCandidate]:
        """
        Merge overlapping clip candidates.
        If two candidates overlap by more than 5 seconds, retain the wider boundary
        and the highest score.
        """
        if not candidates:
            return []

        # Sort by start time
        sorted_cands = sorted(candidates, key=lambda c: c.start)
        merged = [sorted_cands[0]]

        for curr in sorted_cands[1:]:
            prev = merged[-1]

            # Check for overlap or close proximity (< 5s gap)
            if curr.start <= prev.end + 5.0:
                # Merge into one
                new_start = min(prev.start, curr.start)
                new_end = max(prev.end, curr.end)
                new_score = max(prev.score, curr.score)

                # Keep the better hook text and reason
                better_cand = curr if curr.score > prev.score else prev
                combined_seg_ids = sorted(list(set(prev.segment_ids + curr.segment_ids)))

                merged[-1] = ClipCandidate(
                    start=new_start,
                    end=new_end,
                    score=new_score,
                    hook_text=better_cand.hook_text,
                    reason=better_cand.reason,
                    segment_ids=combined_seg_ids,
                    scores_breakdown=better_cand.scores_breakdown,
                )
            else:
                merged.append(curr)

        return merged
