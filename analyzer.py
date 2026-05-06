"""
Content Analyzer Module
Analyzes transcript to find the most compelling, hook-worthy moments.

Scoring dimensions:
1. TF-IDF importance — sentences that contain rare, meaningful words
2. Quote detection — direct quotes, emphasis patterns
3. Hook keywords — trigger words that signal high-value content
4. Sentiment intensity — emotionally charged segments
5. Position bonus — opening hooks and climax positioning

Outputs ranked clip candidates with smart boundary detection.
"""
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional

from transcriber import Transcript, TranscriptSegment
from config import AnalyzerConfig

logger = logging.getLogger(__name__)

# ─── Stopwords (common English words to ignore in TF-IDF) ───
STOPWORDS = set("""
a an the is are was were be been being have has had do does did will would
shall should may might can could must need dare ought i me my mine myself
we us our ours ourselves you your yours yourself yourselves he him his
himself she her hers herself it its itself they them their theirs themselves
what which who whom this that these those am is are was were be been being
have has had having do does did doing a an the and but if or because as
until while of at by for with about against between through during before
after above below to from up down in out on off over under again further
then once here there when where why how all both each few more most other
some such no nor not only own same so than too very s t can will just don
should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn
ma mightn mustn needn shan shouldn wasn weren won wouldn um uh like yeah
yes no okay ok well so just really actually know think mean right going
gonna wanna got get got let say said says thing things go went come came
""".split())


@dataclass
class ClipCandidate:
    """A candidate clip with score and metadata."""
    start: float
    end: float
    score: float
    hook_text: str  # the key sentence(s) that triggered this clip
    reason: str     # why this was flagged as a hook
    segment_ids: list = field(default_factory=list)
    scores_breakdown: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


class ContentAnalyzer:
    """Analyzes transcript content to find hook-worthy moments."""

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()
        self._hook_pattern = re.compile(
            r'|'.join(re.escape(kw) for kw in self.config.hook_keywords),
            re.IGNORECASE
        )

    def analyze(self, transcript: Transcript) -> list:
        """
        Analyze transcript and return ranked list of ClipCandidates.

        Pipeline:
        1. Group segments into sliding windows
        2. Score each window on multiple dimensions
        3. Merge overlapping high-score windows
        4. Return top N candidates sorted by score
        """
        if not transcript.segments:
            logger.warning("Empty transcript, no clips to generate.")
            return []

        logger.info(f"Analyzing {len(transcript.segments)} transcript segments...")

        # Step 1: Build segment windows (groups of consecutive segments)
        windows = self._build_windows(transcript.segments, transcript.duration)
        logger.info(f"Built {len(windows)} analysis windows")

        # Step 2: Compute TF-IDF scores for the entire transcript
        tfidf_scores = self._compute_tfidf(transcript.segments)

        # Step 3: Score each window
        scored_windows = []
        for window in windows:
            score, breakdown, reason = self._score_window(
                window, tfidf_scores, transcript.duration
            )
            if score >= self.config.min_hook_score:
                text = " ".join(seg.text for seg in window["segments"])
                scored_windows.append(ClipCandidate(
                    start=window["start"],
                    end=window["end"],
                    score=round(score, 4),
                    hook_text=text[:300],
                    reason=reason,
                    segment_ids=[s.id for s in window["segments"]],
                    scores_breakdown=breakdown,
                ))

        logger.info(f"{len(scored_windows)} windows passed score threshold ({self.config.min_hook_score})")

        # Step 4: Merge overlapping candidates
        merged = self._merge_overlapping(scored_windows)
        logger.info(f"{len(merged)} candidates after merging overlaps")

        # Step 5: Sort by score and take top N
        merged.sort(key=lambda c: c.score, reverse=True)
        top_clips = merged[:self.config.max_clips]

        # Sort final output by timestamp
        top_clips.sort(key=lambda c: c.start)

        logger.info(f"Final clip candidates: {len(top_clips)}")
        for i, clip in enumerate(top_clips):
            logger.info(
                f"  Clip {i+1}: {clip.start:.1f}s - {clip.end:.1f}s "
                f"(score={clip.score:.3f}, reason={clip.reason})"
            )

        return top_clips

    def _build_windows(self, segments: list, total_duration: float) -> list:
        """
        Build overlapping windows of segments.
        Each window spans 2-6 consecutive segments to capture natural thought groups.
        """
        windows = []
        n = len(segments)

        for window_size in [2, 3, 4, 5, 6]:
            for i in range(n - window_size + 1):
                window_segs = segments[i:i + window_size]
                start = window_segs[0].start
                end = window_segs[-1].end
                duration = end - start

                # Skip windows that are too short or too long
                if duration < 10 or duration > 90:
                    continue

                windows.append({
                    "segments": window_segs,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "text": " ".join(s.text for s in window_segs),
                })

        return windows

    def _compute_tfidf(self, segments: list) -> dict:
        """
        Compute TF-IDF scores for each segment.
        Returns dict mapping segment_id -> tfidf_score (normalized 0-1).
        """
        # Tokenize each segment
        doc_tokens = {}
        for seg in segments:
            tokens = self._tokenize(seg.text)
            doc_tokens[seg.id] = tokens

        # Document frequency
        total_docs = len(segments)
        df = Counter()
        for tokens in doc_tokens.values():
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1

        # Compute TF-IDF per segment
        tfidf_scores = {}
        max_score = 0

        for seg_id, tokens in doc_tokens.items():
            if not tokens:
                tfidf_scores[seg_id] = 0
                continue

            tf = Counter(tokens)
            score = 0
            for token, count in tf.items():
                tf_val = count / len(tokens)
                idf_val = math.log((total_docs + 1) / (df[token] + 1)) + 1
                score += tf_val * idf_val

            tfidf_scores[seg_id] = score
            max_score = max(max_score, score)

        # Normalize to 0-1
        if max_score > 0:
            for seg_id in tfidf_scores:
                tfidf_scores[seg_id] /= max_score

        return tfidf_scores

    def _score_window(self, window: dict, tfidf_scores: dict,
                      total_duration: float) -> tuple:
        """
        Score a window across all dimensions.
        Returns (total_score, breakdown_dict, primary_reason).
        """
        text = window["text"]
        segments = window["segments"]
        cfg = self.config

        # 1. TF-IDF score (average of segment scores)
        seg_tfidf = [tfidf_scores.get(s.id, 0) for s in segments]
        tfidf_score = sum(seg_tfidf) / len(seg_tfidf) if seg_tfidf else 0

        # 2. Quote detection score
        quote_score = self._score_quotes(text)

        # 3. Hook keyword score
        keyword_score = self._score_keywords(text)

        # 4. Sentiment intensity score
        sentiment_score = self._score_sentiment(text)

        # 5. Position score (opening and climax moments score higher)
        mid_time = (window["start"] + window["end"]) / 2
        position_score = self._score_position(mid_time, total_duration)

        # Weighted combination
        total = (
            cfg.tfidf_weight * tfidf_score +
            cfg.quote_weight * quote_score +
            cfg.keyword_weight * keyword_score +
            cfg.sentiment_weight * sentiment_score +
            cfg.position_weight * position_score
        )

        breakdown = {
            "tfidf": round(tfidf_score, 4),
            "quote": round(quote_score, 4),
            "keyword": round(keyword_score, 4),
            "sentiment": round(sentiment_score, 4),
            "position": round(position_score, 4),
        }

        # Determine primary reason
        score_map = {
            "high_importance_content": tfidf_score * cfg.tfidf_weight,
            "contains_quotes": quote_score * cfg.quote_weight,
            "hook_keywords_detected": keyword_score * cfg.keyword_weight,
            "emotionally_charged": sentiment_score * cfg.sentiment_weight,
            "strategic_position": position_score * cfg.position_weight,
        }
        reason = max(score_map, key=score_map.get)

        return total, breakdown, reason

    def _score_quotes(self, text: str) -> float:
        """Detect quoted speech, emphasis, and direct address patterns."""
        score = 0.0
        lower = text.lower()

        # Direct quotes
        quote_count = text.count('"') // 2 + text.count("'") // 2
        if quote_count > 0:
            score += min(0.4, quote_count * 0.2)

        # Emphasis patterns: "THIS is what I mean", rhetorical questions
        if re.search(r'\b[A-Z]{2,}\b', text):
            score += 0.2

        # Rhetorical questions
        question_count = text.count('?')
        if question_count > 0:
            score += min(0.3, question_count * 0.15)

        # Direct address patterns
        direct_patterns = [
            r'\byou need to\b', r'\byou have to\b', r'\byou should\b',
            r'\blet me tell you\b', r'\bhere\'s what\b', r'\bthe thing is\b',
            r'\bi\'m going to show\b', r'\bwatch this\b', r'\bcheck this out\b',
            r'\blisten\b', r'\bpay attention\b',
        ]
        for pattern in direct_patterns:
            if re.search(pattern, lower):
                score += 0.15

        return min(1.0, score)

    def _score_keywords(self, text: str) -> float:
        """Score based on presence of hook/trigger keywords."""
        matches = self._hook_pattern.findall(text)
        if not matches:
            return 0.0
        # Diminishing returns: first few keywords matter most
        count = len(matches)
        unique_count = len(set(m.lower() for m in matches))
        return min(1.0, (unique_count * 0.2) + (count * 0.05))

    def _score_sentiment(self, text: str) -> float:
        """
        Score emotional intensity using lexicon-free heuristics.
        High intensity (positive or negative) = good hook material.
        """
        score = 0.0
        lower = text.lower()

        # Exclamation marks signal intensity
        excl_count = text.count('!')
        score += min(0.3, excl_count * 0.1)

        # Superlatives and absolutes
        intensity_words = [
            'best', 'worst', 'greatest', 'biggest', 'most', 'least',
            'never', 'always', 'every', 'none', 'everything', 'nothing',
            'amazing', 'terrible', 'incredible', 'unbelievable', 'insane',
            'absolutely', 'completely', 'totally', 'literally', 'extremely',
            'love', 'hate', 'obsessed', 'destroyed', 'killed', 'crushed',
            'exploded', 'massive', 'tiny', 'huge', 'revolutionary',
        ]
        for word in intensity_words:
            if word in lower:
                score += 0.1

        # Contrast and pivot language
        contrast_patterns = [
            r'\bbut\b', r'\bhowever\b', r'\bon the other hand\b',
            r'\bactually\b', r'\bin reality\b', r'\bthe truth is\b',
            r'\bcontrary to\b', r'\binstead\b', r'\bplot twist\b',
        ]
        for pattern in contrast_patterns:
            if re.search(pattern, lower):
                score += 0.15

        return min(1.0, score)

    def _score_position(self, mid_time: float, total_duration: float) -> float:
        """
        Score based on position in the video.
        Openings (first 15%) and climax zone (60-80%) score higher.
        """
        if total_duration == 0:
            return 0.5

        relative_pos = mid_time / total_duration

        # Opening hook zone (0-15%)
        if relative_pos < 0.15:
            return 0.8 + (0.2 * (1 - relative_pos / 0.15))

        # Climax zone (60-80%)
        if 0.60 <= relative_pos <= 0.80:
            return 0.7

        # Closing remarks (last 10%)
        if relative_pos > 0.90:
            return 0.5

        # Middle content
        return 0.3

    def _merge_overlapping(self, candidates: list) -> list:
        """Merge overlapping clip candidates, keeping the higher score."""
        if not candidates:
            return []

        sorted_cands = sorted(candidates, key=lambda c: c.start)
        merged = [sorted_cands[0]]

        for current in sorted_cands[1:]:
            prev = merged[-1]
            # Check overlap or close proximity
            if current.start <= prev.end + 5:  # 5s gap tolerance
                # Merge: extend time range, keep higher score
                if current.score > prev.score:
                    merged[-1] = ClipCandidate(
                        start=min(prev.start, current.start),
                        end=max(prev.end, current.end),
                        score=max(prev.score, current.score),
                        hook_text=current.hook_text,
                        reason=current.reason,
                        segment_ids=list(set(prev.segment_ids + current.segment_ids)),
                        scores_breakdown=current.scores_breakdown,
                    )
                else:
                    merged[-1] = ClipCandidate(
                        start=min(prev.start, current.start),
                        end=max(prev.end, current.end),
                        score=prev.score,
                        hook_text=prev.hook_text,
                        reason=prev.reason,
                        segment_ids=list(set(prev.segment_ids + current.segment_ids)),
                        scores_breakdown=prev.scores_breakdown,
                    )
            else:
                merged.append(current)

        return merged

    def _tokenize(self, text: str) -> list:
        """Simple tokenizer: lowercase, remove punctuation, filter stopwords."""
        tokens = re.findall(r'[a-z]+', text.lower())
        return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    def save_analysis(self, candidates: list, path: str):
        """Save analysis results to JSON."""
        data = [c.to_dict() for c in candidates]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Analysis saved to {path}")
