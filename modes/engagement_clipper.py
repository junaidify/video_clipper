"""
Engagement-Based Clipper for Medium-Length Videos (5-20 min segments)

Analyzes long videos (30min+) to find the most entertaining, high-engagement
segments worth extracting as standalone clips. Unlike the short-form Smart Clips
(15-60s), this targets medium-length content suitable for YouTube, compilations,
or highlight reels.

Scoring dimensions:
1. Dialogue density — segments with more speech per second = more engaging
2. Emotional intensity — sentiment peaks, exclamation, questions, emphasis
3. Topic coherence — segments that form a complete thought/story arc
4. Content quality — unique vocabulary, information density (TF-IDF)
5. Pacing — variation in sentence length, speaker energy changes
6. LLM analysis — when available, uses AI to identify the most entertaining moments

Output: ranked segments with scores + FFmpeg extraction with all post-processing.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EngagementConfig:
    """Configuration for engagement-based medium-length clipping."""
    # Duration bounds for extracted segments (seconds)
    min_segment_duration: int = 300    # 5 minutes
    max_segment_duration: int = 1200   # 20 minutes
    target_segment_duration: int = 600  # 10 minutes ideal
    # How many segments to extract
    max_segments: int = 5
    # Minimum score (0-1) to qualify
    min_engagement_score: float = 0.35
    # Analysis window: how many transcript segments to group per analysis unit
    window_segments: int = 20  # ~2-3 min of speech per window
    window_stride: int = 5    # overlap between windows
    # Scoring weights
    dialogue_weight: float = 0.20
    emotion_weight: float = 0.25
    coherence_weight: float = 0.15
    quality_weight: float = 0.15
    pacing_weight: float = 0.10
    llm_weight: float = 0.15
    # Post-processing
    crop_vertical: bool = True
    video_quality: int = 23
    # Use LLM for segment identification (if API key available)
    use_llm: bool = True


@dataclass
class EngagementSegment:
    """A scored segment candidate."""
    start: float
    end: float
    score: float
    title_hint: str       # suggested title based on content
    summary: str          # brief description of what happens
    reason: str           # primary scoring reason
    scores_breakdown: dict = field(default_factory=dict)
    segment_ids: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "duration": round(self.duration, 1),
            "score": round(self.score, 4),
            "title_hint": self.title_hint,
            "summary": self.summary,
            "reason": self.reason,
            "scores_breakdown": self.scores_breakdown,
        }


# ─── Stopwords for quality scoring ───
_STOPWORDS = set("""
a an the is are was were be been being have has had do does did will would
shall should may might can could must i me my we us our you your he him his
she her it its they them their what which who whom this that these those
am and but if or because as until while of at by for with about against
between through during before after above below to from up down in out on
off over under again further then once here there when where why how all
both each few more most other some such no nor not only own same so than
too very um uh like yeah yes no okay ok well just really actually know
think mean right going gonna wanna got get let say said says thing things
go went come came
""".split())


class EngagementAnalyzer:
    """Analyzes transcript to find high-engagement medium-length segments."""

    def __init__(self, config: Optional[EngagementConfig] = None):
        self.config = config or EngagementConfig()

    def analyze(self, transcript, llm_segments: list = None) -> list:
        """
        Analyze transcript and return ranked EngagementSegments.

        Args:
            transcript: Transcript object with segments
            llm_segments: Optional pre-analyzed segments from LLM

        Returns:
            List of EngagementSegment sorted by score descending
        """
        if not transcript or not transcript.segments:
            return []

        segs = transcript.segments
        total_duration = transcript.duration
        cfg = self.config

        logger.info(
            f"Engagement analysis: {len(segs)} segments, "
            f"{total_duration:.0f}s video, target {cfg.min_segment_duration}-{cfg.max_segment_duration}s clips"
        )

        # Build wide windows for analysis
        windows = self._build_analysis_windows(segs, total_duration)
        logger.info(f"Built {len(windows)} analysis windows")

        # Compute TF-IDF for quality scoring
        tfidf = self._compute_tfidf(segs)

        # Score each window
        scored = []
        for w in windows:
            score, breakdown, reason = self._score_window(w, tfidf, total_duration)
            if score >= cfg.min_engagement_score:
                # Generate title hint from the most distinctive content
                title_hint = self._extract_title_hint(w)
                summary = self._extract_summary(w)
                scored.append(EngagementSegment(
                    start=w["start"],
                    end=w["end"],
                    score=round(score, 4),
                    title_hint=title_hint,
                    summary=summary,
                    reason=reason,
                    scores_breakdown=breakdown,
                    segment_ids=[s.id for s in w["segments"]],
                ))

        # Merge overlapping segments
        merged = self._merge_overlapping(scored)

        # If LLM segments provided, boost matching windows
        if llm_segments:
            merged = self._apply_llm_boost(merged, llm_segments)

        # Sort by score, take top N
        merged.sort(key=lambda s: s.score, reverse=True)
        top = merged[:cfg.max_segments]

        # Sort final output by timestamp
        top.sort(key=lambda s: s.start)

        logger.info(f"Engagement analysis: {len(top)} segments selected")
        for i, seg in enumerate(top):
            logger.info(
                f"  Segment {i+1}: {seg.start:.0f}s-{seg.end:.0f}s "
                f"({seg.duration:.0f}s, score={seg.score:.3f}, {seg.reason})"
            )

        return top

    def _build_analysis_windows(self, segments: list, total_duration: float) -> list:
        """Build wide overlapping windows for medium-length segment analysis."""
        cfg = self.config
        windows = []
        n = len(segments)

        # Slide through transcript with configurable stride
        for i in range(0, n - cfg.window_segments + 1, cfg.window_stride):
            # Try multiple window sizes to find natural boundaries
            for extra in [0, 5, 10, 15]:
                end_idx = min(i + cfg.window_segments + extra, n)
                window_segs = segments[i:end_idx]

                if not window_segs:
                    continue

                start = window_segs[0].start
                end = window_segs[-1].end
                duration = end - start

                # Only consider windows within our target range
                if duration < cfg.min_segment_duration * 0.7:
                    continue
                if duration > cfg.max_segment_duration * 1.2:
                    break  # no point going wider

                text = " ".join(s.text for s in window_segs)
                windows.append({
                    "segments": window_segs,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "text": text,
                })

        return windows

    def _score_window(self, window: dict, tfidf: dict, total_duration: float) -> tuple:
        """Score a window across all engagement dimensions."""
        text = window["text"]
        segments = window["segments"]
        duration = window["duration"]
        cfg = self.config

        # 1. Dialogue density (speech seconds / total seconds)
        speech_time = sum(s.end - s.start for s in segments)
        dialogue_score = min(1.0, speech_time / duration) if duration > 0 else 0

        # 2. Emotional intensity
        emotion_score = self._score_emotion(text, segments)

        # 3. Topic coherence (how focused is this section on one topic)
        coherence_score = self._score_coherence(segments)

        # 4. Content quality (TF-IDF uniqueness)
        seg_tfidf = [tfidf.get(s.id, 0) for s in segments]
        quality_score = (sum(seg_tfidf) / len(seg_tfidf)) if seg_tfidf else 0

        # 5. Pacing (variation in segment lengths = dynamic pacing)
        pacing_score = self._score_pacing(segments)

        # Weighted combination
        total = (
            cfg.dialogue_weight * dialogue_score +
            cfg.emotion_weight * emotion_score +
            cfg.coherence_weight * coherence_score +
            cfg.quality_weight * quality_score +
            cfg.pacing_weight * pacing_score
        )

        breakdown = {
            "dialogue": round(dialogue_score, 3),
            "emotion": round(emotion_score, 3),
            "coherence": round(coherence_score, 3),
            "quality": round(quality_score, 3),
            "pacing": round(pacing_score, 3),
        }

        # Determine primary reason
        score_map = {
            "high_dialogue_density": dialogue_score * cfg.dialogue_weight,
            "emotional_intensity": emotion_score * cfg.emotion_weight,
            "topic_coherence": coherence_score * cfg.coherence_weight,
            "unique_content": quality_score * cfg.quality_weight,
            "dynamic_pacing": pacing_score * cfg.pacing_weight,
        }
        reason = max(score_map, key=score_map.get)

        return total, breakdown, reason

    def _score_emotion(self, text: str, segments: list) -> float:
        """Score emotional intensity of the segment."""
        score = 0.0
        lower = text.lower()

        # Exclamation marks
        score += min(0.2, text.count('!') * 0.02)

        # Question marks (engagement drivers)
        score += min(0.15, text.count('?') * 0.02)

        # Intensity words
        intensity = [
            'amazing', 'incredible', 'unbelievable', 'shocking', 'insane',
            'best', 'worst', 'never', 'always', 'every', 'nothing',
            'love', 'hate', 'beautiful', 'terrible', 'perfect',
            'impossible', 'absolutely', 'completely', 'literally',
            'destroyed', 'killed', 'crushed', 'exploded', 'revolutionary',
            'pyaar', 'mohabbat', 'dil', 'ishq', 'pagal', 'maut',
            'zindagi', 'sapna', 'dard', 'khushi', 'rona',
        ]
        for w in intensity:
            if w in lower:
                score += 0.04

        # Contrast/pivot language (story turning points)
        pivots = [
            r'\bbut\b', r'\bhowever\b', r'\bactually\b', r'\bin reality\b',
            r'\bthe truth\b', r'\bplot twist\b', r'\bwait\b', r'\bsurprise\b',
            r'\blekin\b', r'\bmagar\b', r'\bsach\b', r'\basli\b',
        ]
        for p in pivots:
            if re.search(p, lower):
                score += 0.05

        # Direct address (engagement)
        directs = [
            r'\byou need\b', r'\byou have to\b', r'\blisten\b',
            r'\bwatch this\b', r'\bpay attention\b', r'\blook at\b',
            r'\bdekho\b', r'\bsuno\b', r'\bsamjho\b',
        ]
        for d in directs:
            if re.search(d, lower):
                score += 0.04

        return min(1.0, score)

    def _score_coherence(self, segments: list) -> float:
        """Score how topically focused/coherent a segment is."""
        if len(segments) < 3:
            return 0.5

        # Tokenize each segment, compute vocabulary overlap between consecutive groups
        def _tokens(text):
            return set(re.findall(r'[a-z]+', text.lower())) - _STOPWORDS

        # Split into thirds and measure vocabulary overlap
        n = len(segments)
        third = max(1, n // 3)
        t1 = _tokens(" ".join(s.text for s in segments[:third]))
        t2 = _tokens(" ".join(s.text for s in segments[third:2*third]))
        t3 = _tokens(" ".join(s.text for s in segments[2*third:]))

        if not t1 or not t2 or not t3:
            return 0.3

        # Jaccard similarity between consecutive thirds
        overlap_12 = len(t1 & t2) / max(len(t1 | t2), 1)
        overlap_23 = len(t2 & t3) / max(len(t2 | t3), 1)
        avg_overlap = (overlap_12 + overlap_23) / 2

        # Higher overlap = more coherent = better self-contained segment
        return min(1.0, avg_overlap * 3)

    def _score_pacing(self, segments: list) -> float:
        """Score pacing variation — dynamic content scores higher."""
        if len(segments) < 4:
            return 0.5

        # Measure variance in segment durations
        durations = [s.end - s.start for s in segments]
        mean_dur = sum(durations) / len(durations)
        if mean_dur <= 0:
            return 0.3

        variance = sum((d - mean_dur) ** 2 for d in durations) / len(durations)
        cv = math.sqrt(variance) / mean_dur  # coefficient of variation

        # Moderate variation = good pacing (not monotone, not chaotic)
        # cv of 0.3-0.7 is ideal
        if 0.3 <= cv <= 0.7:
            return 0.9
        elif 0.15 <= cv <= 1.0:
            return 0.6
        else:
            return 0.3

    def _compute_tfidf(self, segments: list) -> dict:
        """Compute TF-IDF scores for quality scoring."""
        doc_tokens = {}
        for seg in segments:
            tokens = [t for t in re.findall(r'[a-z]+', seg.text.lower())
                      if t not in _STOPWORDS and len(t) > 2]
            doc_tokens[seg.id] = tokens

        total_docs = len(segments)
        df = Counter()
        for tokens in doc_tokens.values():
            for t in set(tokens):
                df[t] += 1

        scores = {}
        max_score = 0
        for seg_id, tokens in doc_tokens.items():
            if not tokens:
                scores[seg_id] = 0
                continue
            tf = Counter(tokens)
            s = sum(
                (c / len(tokens)) * (math.log((total_docs + 1) / (df[t] + 1)) + 1)
                for t, c in tf.items()
            )
            scores[seg_id] = s
            max_score = max(max_score, s)

        if max_score > 0:
            for k in scores:
                scores[k] /= max_score

        return scores

    def _merge_overlapping(self, segments: list) -> list:
        """Merge overlapping segment candidates."""
        if not segments:
            return []

        sorted_segs = sorted(segments, key=lambda s: s.start)
        merged = [sorted_segs[0]]

        for curr in sorted_segs[1:]:
            prev = merged[-1]
            # Overlap if current starts before previous ends (with 30s tolerance)
            if curr.start <= prev.end + 30:
                if curr.score > prev.score:
                    merged[-1] = EngagementSegment(
                        start=min(prev.start, curr.start),
                        end=max(prev.end, curr.end),
                        score=max(prev.score, curr.score),
                        title_hint=curr.title_hint,
                        summary=curr.summary,
                        reason=curr.reason,
                        scores_breakdown=curr.scores_breakdown,
                        segment_ids=list(set(prev.segment_ids + curr.segment_ids)),
                    )
                else:
                    merged[-1] = EngagementSegment(
                        start=min(prev.start, curr.start),
                        end=max(prev.end, curr.end),
                        score=prev.score,
                        title_hint=prev.title_hint,
                        summary=prev.summary,
                        reason=prev.reason,
                        scores_breakdown=prev.scores_breakdown,
                        segment_ids=list(set(prev.segment_ids + curr.segment_ids)),
                    )
            else:
                merged.append(curr)

        # Enforce duration limits
        cfg = self.config
        result = []
        for seg in merged:
            if seg.duration > cfg.max_segment_duration:
                # Trim to max from the center
                center = (seg.start + seg.end) / 2
                half = cfg.max_segment_duration / 2
                seg = EngagementSegment(
                    start=max(0, center - half),
                    end=center + half,
                    score=seg.score,
                    title_hint=seg.title_hint,
                    summary=seg.summary,
                    reason=seg.reason,
                    scores_breakdown=seg.scores_breakdown,
                    segment_ids=seg.segment_ids,
                )
            if seg.duration >= cfg.min_segment_duration * 0.7:
                result.append(seg)

        return result

    def _apply_llm_boost(self, segments: list, llm_segments: list) -> list:
        """Boost scores of segments that overlap with LLM-identified highlights."""
        for seg in segments:
            for llm in llm_segments:
                llm_start = llm.get('start', 0)
                llm_end = llm.get('end', 0)
                llm_score = llm.get('score', 0.5)
                # Check overlap
                overlap_start = max(seg.start, llm_start)
                overlap_end = min(seg.end, llm_end)
                if overlap_end > overlap_start:
                    overlap_ratio = (overlap_end - overlap_start) / seg.duration
                    boost = self.config.llm_weight * llm_score * overlap_ratio
                    seg.score = min(1.0, seg.score + boost)
                    seg.scores_breakdown['llm_boost'] = round(boost, 3)
                    if llm.get('reason'):
                        seg.reason = f"llm_{llm['reason']}"
        return segments

    def _extract_title_hint(self, window: dict) -> str:
        """Extract a title hint from the most distinctive content in the window."""
        text = window["text"]
        # Find the most information-dense sentence
        sentences = re.split(r'[.!?]+', text)
        if not sentences:
            return "Highlight"

        # Pick the sentence with the most unique words
        best = max(sentences, key=lambda s: len(
            set(re.findall(r'[a-z]+', s.lower())) - _STOPWORDS
        ))
        best = best.strip()
        if len(best) > 60:
            best = best[:57] + "..."
        return best or "Highlight"

    def _extract_summary(self, window: dict) -> str:
        """Generate a brief summary of the segment content."""
        text = window["text"]
        words = text.split()
        if len(words) <= 20:
            return text
        # First 15 words + last 5 words
        return " ".join(words[:15]) + " ... " + " ".join(words[-5:])


def analyze_with_llm_for_engagement(transcript, config: EngagementConfig = None) -> list:
    """
    Use LLM to identify the most entertaining segments for medium-length clips.
    Returns list of dicts: [{start, end, score, reason, title_hint}, ...]
    """
    config = config or EngagementConfig()

    try:
        from core.llm_analyzer import LLMConfig

        llm_config = LLMConfig.from_env()
        if not llm_config.has_any_key():
            return []

        # Build transcript text with timestamps
        lines = []
        for seg in transcript.segments:
            lines.append(f"[{seg.start:.0f}s-{seg.end:.0f}s] {seg.text}")
        transcript_text = "\n".join(lines)

        # Truncate if too long (keep first and last portions)
        if len(transcript_text) > 8000:
            half = 3500
            transcript_text = transcript_text[:half] + "\n...[middle truncated]...\n" + transcript_text[-half:]

        prompt = f"""Analyze this video transcript and identify the {config.max_segments} most ENTERTAINING,
engaging segments worth extracting as standalone clips ({config.min_segment_duration//60}-{config.max_segment_duration//60} minutes each).

Look for:
- Emotional peaks (arguments, confessions, revelations, humor, romance)
- Complete story arcs (setup → conflict → resolution)
- High dialogue density with multiple speakers
- Dramatic tension or suspense
- Memorable quotes or key moments
- Content that works as a self-contained clip

TRANSCRIPT:
{transcript_text}

VIDEO DURATION: {transcript.duration:.0f} seconds

Return ONLY valid JSON array:
[
  {{
    "start": <seconds>,
    "end": <seconds>,
    "score": <0.0-1.0>,
    "reason": "<why this segment is entertaining>",
    "title_hint": "<suggested clip title, 5-8 words>"
  }}
]"""

        # Try providers in order
        result_text = None
        if llm_config.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=llm_config.gemini_api_key)
                model = genai.GenerativeModel(llm_config.gemini_model)
                resp = model.generate_content(prompt)
                result_text = resp.text.strip()
            except Exception as e:
                logger.warning(f"Gemini engagement analysis failed: {e}")

        if not result_text and llm_config.nvidia_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=llm_config.nvidia_api_key,
                )
                resp = client.chat.completions.create(
                    model=llm_config.nvidia_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=2048,
                )
                result_text = resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"NVIDIA engagement analysis failed: {e}")

        if not result_text and llm_config.groq_api_key:
            try:
                from groq import Groq
                client = Groq(api_key=llm_config.groq_api_key)
                resp = client.chat.completions.create(
                    model=llm_config.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=2048,
                )
                result_text = resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Groq engagement analysis failed: {e}")

        if not result_text:
            return []

        # Parse JSON
        import json
        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[1] if "\n" in result_text else result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3].strip()

        segments = json.loads(result_text)
        logger.info(f"LLM identified {len(segments)} entertainment segments")
        return segments

    except Exception as e:
        logger.warning(f"LLM engagement analysis failed: {e}")
        return []
