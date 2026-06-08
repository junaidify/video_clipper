"""
Shared pytest fixtures for Video Auto-Clipper tests.
"""
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.transcriber import Transcript, TranscriptSegment
from core.analyzer import ClipCandidate
from config import PipelineConfig, TranscriberConfig, AnalyzerConfig, ClipperConfig


@pytest.fixture
def mock_transcript():
    """A realistic mock transcript with 10 segments spanning 300 seconds."""
    segments = [
        TranscriptSegment(id=0, start=0.0, end=15.0,
                          text="Here's the thing most people don't know about success.",
                          words=[]),
        TranscriptSegment(id=1, start=15.0, end=35.0,
                          text="The number one mistake everyone makes is thinking they need more time.",
                          words=[]),
        TranscriptSegment(id=2, start=35.0, end=55.0,
                          text="Studies show that the top 1 percent work differently not harder.",
                          words=[]),
        TranscriptSegment(id=3, start=55.0, end=80.0,
                          text="Let me tell you a true story about how I lost everything.",
                          words=[]),
        TranscriptSegment(id=4, start=80.0, end=110.0,
                          text="I was sitting in my apartment and then the phone rang.",
                          words=[]),
        TranscriptSegment(id=5, start=110.0, end=140.0,
                          text="That changes everything about the way you think about money.",
                          words=[]),
        TranscriptSegment(id=6, start=140.0, end=170.0,
                          text="Step one is to build a framework for daily execution.",
                          words=[]),
        TranscriptSegment(id=7, start=170.0, end=200.0,
                          text="The data shows a 300 percent improvement in just 90 days.",
                          words=[]),
        TranscriptSegment(id=8, start=200.0, end=250.0,
                          text="Oh my god wait a minute that just blew my mind!",
                          words=[]),
        TranscriptSegment(id=9, start=250.0, end=300.0,
                          text="Remember this is the most important lesson of your life.",
                          words=[]),
    ]
    full_text = " ".join(s.text for s in segments)
    return Transcript(
        segments=segments,
        full_text=full_text,
        language="en",
        duration=300.0,
    )


@pytest.fixture
def empty_transcript():
    """An empty transcript with no segments."""
    return Transcript(segments=[], full_text="", language="en", duration=0.0)


@pytest.fixture
def mock_clip_candidates():
    """A list of mock ClipCandidate objects."""
    return [
        ClipCandidate(start=10.0, end=40.0, score=0.85,
                      hook_text="Here's the thing most people don't know",
                      reason="hook_keywords_detected", segment_ids=[0, 1]),
        ClipCandidate(start=80.0, end=120.0, score=0.72,
                      hook_text="Let me tell you a true story",
                      reason="viral_pattern_match", segment_ids=[3, 4]),
        ClipCandidate(start=200.0, end=260.0, score=0.65,
                      hook_text="Oh my god that blew my mind",
                      reason="emotionally_charged", segment_ids=[8, 9]),
    ]


@pytest.fixture
def mock_video_path(tmp_path):
    """A fake video file path (file does not actually exist)."""
    return str(tmp_path / "test_video.mp4")


@pytest.fixture
def flask_client():
    """Flask test client with mocked heavy dependencies."""
    with patch("media.library.VideoLibrary"), \
         patch("training.trainer.PatternTrainer"):
        from app import app as flask_app
        flask_app.config["TESTING"] = True
        with flask_app.test_client() as client:
            yield client
