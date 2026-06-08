"""Tests for core.clipper module (all FFmpeg calls mocked)."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.analyzer import ClipCandidate
from config import ClipperConfig


# Patch ffmpeg verification globally for all tests in this module
@pytest.fixture(autouse=True)
def mock_ffmpeg():
    with patch("core.clipper.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        yield mock_run


def _make_clipper(config=None):
    from core.clipper import VideoClipper
    return VideoClipper(config)


class TestAdjustBoundaries:
    def test_adds_padding(self):
        clipper = _make_clipper()
        candidate = ClipCandidate(start=30.0, end=50.0, score=0.8,
                                  hook_text="test", reason="test")
        start, end = clipper._adjust_boundaries(candidate, video_duration=300.0)
        assert start == 27.0   # 30 - 3
        assert end == 55.0     # 50 + 5

    def test_clamps_to_zero(self):
        clipper = _make_clipper()
        candidate = ClipCandidate(start=1.0, end=20.0, score=0.8,
                                  hook_text="test", reason="test")
        start, end = clipper._adjust_boundaries(candidate, video_duration=300.0)
        assert start >= 0.0

    def test_clamps_to_video_duration(self):
        clipper = _make_clipper()
        candidate = ClipCandidate(start=290.0, end=300.0, score=0.8,
                                  hook_text="test", reason="test")
        start, end = clipper._adjust_boundaries(candidate, video_duration=300.0)
        assert end <= 300.0

    def test_enforces_min_duration(self):
        cfg = ClipperConfig(min_clip_duration=20)
        clipper = _make_clipper(cfg)
        candidate = ClipCandidate(start=50.0, end=55.0, score=0.8,
                                  hook_text="test", reason="test")
        start, end = clipper._adjust_boundaries(candidate, video_duration=300.0)
        assert (end - start) >= 20

    def test_enforces_max_duration(self):
        cfg = ClipperConfig(max_clip_duration=30)
        clipper = _make_clipper(cfg)
        candidate = ClipCandidate(start=10.0, end=100.0, score=0.8,
                                  hook_text="test", reason="test")
        start, end = clipper._adjust_boundaries(candidate, video_duration=300.0)
        assert (end - start) <= 30


class TestBuildVerticalFilterComplex:
    def test_contains_split_and_overlay(self):
        clipper = _make_clipper()
        fc = clipper._build_vertical_filter_complex(1920, 1080, 30.0)
        assert "[0:v]split=2" in fc
        assert "overlay=" in fc
        assert "[vout]" in fc

    def test_contains_blur(self):
        clipper = _make_clipper()
        fc = clipper._build_vertical_filter_complex(1920, 1080, 30.0)
        assert "gblur" in fc

    def test_contains_fade_when_duration_sufficient(self):
        cfg = ClipperConfig(fade_duration=0.5)
        clipper = _make_clipper(cfg)
        fc = clipper._build_vertical_filter_complex(1920, 1080, 30.0)
        assert "fade=t=in" in fc
        assert "fade=t=out" in fc

    def test_no_fade_when_too_short(self):
        cfg = ClipperConfig(fade_duration=10.0)
        clipper = _make_clipper(cfg)
        fc = clipper._build_vertical_filter_complex(1920, 1080, 5.0)
        assert "fade=" not in fc


class TestBuildFadeFilter:
    def test_returns_filter_string(self):
        cfg = ClipperConfig(fade_duration=0.5)
        clipper = _make_clipper(cfg)
        f = clipper._build_fade_filter(30.0)
        assert "fade=t=in" in f
        assert "fade=t=out" in f

    def test_empty_when_disabled(self):
        cfg = ClipperConfig(fade_duration=0)
        clipper = _make_clipper(cfg)
        assert clipper._build_fade_filter(30.0) == ""

    def test_empty_when_duration_too_short(self):
        cfg = ClipperConfig(fade_duration=5.0)
        clipper = _make_clipper(cfg)
        assert clipper._build_fade_filter(10.0) == ""


class TestClipResult:
    def test_clip_result_dataclass(self):
        from core.clipper import ClipResult
        r = ClipResult(clip_number=1, output_path="/out.mp4", start=0.0,
                       end=30.0, duration=30.0, score=0.8, reason="test",
                       hook_text="hello", success=True)
        assert r.success is True
        assert r.error is None
        assert r.duration == 30.0
