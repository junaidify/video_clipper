"""Tests for core.analyzer module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.analyzer import ContentAnalyzer, ClipCandidate
from config import AnalyzerConfig


class TestTokenize:
    def test_basic_tokenization(self):
        analyzer = ContentAnalyzer()
        tokens = analyzer._tokenize("The secret to amazing success is hard work.")
        assert "secret" in tokens
        assert "amazing" in tokens
        assert "success" in tokens
        # Stopwords removed
        assert "the" not in tokens
        assert "to" not in tokens
        assert "is" not in tokens

    def test_empty_string(self):
        analyzer = ContentAnalyzer()
        assert analyzer._tokenize("") == []

    def test_short_words_filtered(self):
        analyzer = ContentAnalyzer()
        tokens = analyzer._tokenize("I am a go to it")
        # All <= 2 chars or stopwords
        assert len(tokens) == 0

    def test_punctuation_removed(self):
        analyzer = ContentAnalyzer()
        tokens = analyzer._tokenize("Hello, world! This is great.")
        assert "hello" in tokens
        assert "world" in tokens


class TestScoreKeywords:
    def test_no_keywords(self):
        analyzer = ContentAnalyzer()
        score = analyzer._score_keywords("the dog sat on the mat")
        assert score == 0.0

    def test_single_keyword(self):
        analyzer = ContentAnalyzer()
        score = analyzer._score_keywords("This is a secret technique")
        assert score > 0.0

    def test_multiple_keywords_higher_score(self):
        analyzer = ContentAnalyzer()
        single = analyzer._score_keywords("This is a secret")
        multiple = analyzer._score_keywords("This secret hack is amazing and incredible")
        assert multiple > single

    def test_score_capped_at_one(self):
        analyzer = ContentAnalyzer()
        text = " ".join(analyzer.config.hook_keywords[:20])
        score = analyzer._score_keywords(text)
        assert score <= 1.0


class TestScoreSentiment:
    def test_neutral_text(self):
        analyzer = ContentAnalyzer()
        score = analyzer._score_sentiment("The cat sat on the table quietly.")
        assert score == 0.0

    def test_exclamation_marks(self):
        analyzer = ContentAnalyzer()
        score = analyzer._score_sentiment("This is incredible! Amazing! Wow!")
        assert score > 0.0

    def test_intensity_words(self):
        analyzer = ContentAnalyzer()
        score = analyzer._score_sentiment("This is the best and greatest thing ever")
        assert score > 0.0

    def test_contrast_patterns(self):
        analyzer = ContentAnalyzer()
        score = analyzer._score_sentiment("The truth is however that in reality it works")
        assert score > 0.0

    def test_score_capped_at_one(self):
        analyzer = ContentAnalyzer()
        text = "best worst greatest biggest absolutely completely totally literally! ! ! !"
        score = analyzer._score_sentiment(text)
        assert score <= 1.0


class TestScorePosition:
    def test_opening_zone_high(self):
        analyzer = ContentAnalyzer()
        score = analyzer._score_position(5.0, 100.0)
        assert score >= 0.8

    def test_climax_zone(self):
        analyzer = ContentAnalyzer()
        score = analyzer._score_position(70.0, 100.0)
        assert score == 0.7

    def test_middle_zone_low(self):
        analyzer = ContentAnalyzer()
        score = analyzer._score_position(40.0, 100.0)
        assert score == 0.3

    def test_zero_duration(self):
        analyzer = ContentAnalyzer()
        score = analyzer._score_position(10.0, 0.0)
        assert score == 0.5


class TestMergeOverlapping:
    def test_no_overlap(self):
        analyzer = ContentAnalyzer()
        candidates = [
            ClipCandidate(start=0, end=10, score=0.8, hook_text="a", reason="r"),
            ClipCandidate(start=20, end=30, score=0.7, hook_text="b", reason="r"),
        ]
        merged = analyzer._merge_overlapping(candidates)
        assert len(merged) == 2

    def test_overlapping_merged(self):
        analyzer = ContentAnalyzer()
        candidates = [
            ClipCandidate(start=0, end=15, score=0.8, hook_text="a", reason="r"),
            ClipCandidate(start=12, end=25, score=0.9, hook_text="b", reason="r"),
        ]
        merged = analyzer._merge_overlapping(candidates)
        assert len(merged) == 1
        assert merged[0].start == 0
        assert merged[0].end == 25
        assert merged[0].score == 0.9  # keeps higher score

    def test_close_proximity_merged(self):
        """Candidates within 5s gap get merged."""
        analyzer = ContentAnalyzer()
        candidates = [
            ClipCandidate(start=0, end=10, score=0.5, hook_text="a", reason="r"),
            ClipCandidate(start=14, end=25, score=0.6, hook_text="b", reason="r"),
        ]
        merged = analyzer._merge_overlapping(candidates)
        assert len(merged) == 1

    def test_empty_list(self):
        analyzer = ContentAnalyzer()
        assert analyzer._merge_overlapping([]) == []


class TestAnalyze:
    def test_empty_transcript_returns_empty(self, empty_transcript):
        analyzer = ContentAnalyzer()
        result = analyzer.analyze(empty_transcript)
        assert result == []

    def test_returns_clip_candidates(self, mock_transcript):
        analyzer = ContentAnalyzer(AnalyzerConfig(min_hook_score=0.1))
        candidates = analyzer.analyze(mock_transcript)
        assert isinstance(candidates, list)
        for c in candidates:
            assert isinstance(c, ClipCandidate)
            assert c.score >= 0.1

    def test_respects_max_clips(self, mock_transcript):
        analyzer = ContentAnalyzer(AnalyzerConfig(min_hook_score=0.01, max_clips=2))
        candidates = analyzer.analyze(mock_transcript)
        assert len(candidates) <= 2

    def test_candidates_sorted_by_start(self, mock_transcript):
        analyzer = ContentAnalyzer(AnalyzerConfig(min_hook_score=0.01))
        candidates = analyzer.analyze(mock_transcript)
        if len(candidates) >= 2:
            for i in range(len(candidates) - 1):
                assert candidates[i].start <= candidates[i + 1].start
