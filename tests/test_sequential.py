"""Tests for video_clipper.clipping.sequential_splitter module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from video_clipper.clipping.sequential_splitter import compute_split_points, SequentialConfig


class TestSequentialConfig:
    def test_defaults(self):
        cfg = SequentialConfig()
        assert cfg.target_duration == 55
        assert cfg.min_duration == 30
        assert cfg.max_duration == 65
        assert cfg.overlap_seconds == 1.5


class TestComputeSplitPoints:
    def test_zero_duration(self):
        cfg = SequentialConfig()
        assert compute_split_points(0, cfg) == []

    def test_negative_duration(self):
        cfg = SequentialConfig()
        assert compute_split_points(-10, cfg) == []

    def test_short_video_single_reel(self):
        cfg = SequentialConfig(target_duration=60, min_duration=10)
        splits = compute_split_points(45.0, cfg)
        assert len(splits) == 1
        assert splits[0][0] == 0.0
        assert splits[0][1] == 45.0

    def test_covers_entire_video(self):
        cfg = SequentialConfig(target_duration=55, min_duration=20)
        splits = compute_split_points(200.0, cfg)
        assert len(splits) >= 1
        # First reel starts at 0
        assert splits[0][0] == 0.0
        # Last reel reaches (close to) end
        assert splits[-1][1] >= 198.0

    def test_reels_have_overlap(self):
        cfg = SequentialConfig(target_duration=50, overlap_seconds=2.0, min_duration=20)
        splits = compute_split_points(150.0, cfg)
        if len(splits) >= 2:
            # Second reel should start before first reel ends
            assert splits[1][0] < splits[0][1]

    def test_respects_max_duration(self):
        cfg = SequentialConfig(target_duration=55, max_duration=65, min_duration=20)
        splits = compute_split_points(300.0, cfg)
        for start, end in splits:
            assert (end - start) <= 65

    def test_many_reels_for_long_video(self):
        cfg = SequentialConfig(target_duration=55, min_duration=20)
        splits = compute_split_points(600.0, cfg)
        assert len(splits) >= 10  # 600/55 ~ 11

    @pytest.mark.parametrize("duration", [30.0, 55.0, 100.0, 300.0, 1000.0])
    def test_no_negative_times(self, duration):
        cfg = SequentialConfig(target_duration=55, min_duration=10)
        splits = compute_split_points(duration, cfg)
        for start, end in splits:
            assert start >= 0
            assert end > start

    def test_all_splits_are_tuples(self):
        cfg = SequentialConfig()
        splits = compute_split_points(200.0, cfg)
        for item in splits:
            assert isinstance(item, tuple)
            assert len(item) == 2
