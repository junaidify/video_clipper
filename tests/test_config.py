"""Tests for video_clipper.config dataclasses and their defaults."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_clipper.config import TranscriberConfig, AnalyzerConfig, ClipperConfig, PipelineConfig


class TestTranscriberConfig:
    def test_defaults(self):
        cfg = TranscriberConfig()
        assert cfg.model_size == "base"
        assert cfg.language is None
        assert cfg.device == "auto"

    def test_custom_values(self):
        cfg = TranscriberConfig(model_size="large", language="en", device="cuda")
        assert cfg.model_size == "large"
        assert cfg.language == "en"
        assert cfg.device == "cuda"


class TestAnalyzerConfig:
    def test_defaults(self):
        cfg = AnalyzerConfig()
        assert cfg.min_hook_score == 0.4
        assert cfg.max_clips == 10
        assert isinstance(cfg.hook_keywords, list)
        assert len(cfg.hook_keywords) > 0

    def test_weights_sum_to_one(self):
        cfg = AnalyzerConfig()
        total = (cfg.tfidf_weight + cfg.quote_weight + cfg.keyword_weight +
                 cfg.sentiment_weight + cfg.position_weight)
        assert abs(total - 1.0) < 1e-9

    def test_hook_keywords_contains_expected(self):
        cfg = AnalyzerConfig()
        assert "secret" in cfg.hook_keywords
        assert "amazing" in cfg.hook_keywords
        assert "hack" in cfg.hook_keywords


class TestClipperConfig:
    def test_defaults(self):
        cfg = ClipperConfig()
        assert cfg.min_clip_duration == 15
        assert cfg.max_clip_duration == 60
        assert cfg.output_format == "mp4"
        assert cfg.crop_vertical is True
        assert cfg.vertical_width == 1080
        assert cfg.vertical_height == 1920

    def test_min_less_than_max(self):
        cfg = ClipperConfig()
        assert cfg.min_clip_duration < cfg.max_clip_duration

    def test_padding_defaults(self):
        cfg = ClipperConfig()
        assert cfg.pre_hook_padding == 3
        assert cfg.post_hook_padding == 5

    def test_fade_duration(self):
        cfg = ClipperConfig()
        assert cfg.fade_duration == 0.5


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.input_video == ""
        assert cfg.output_dir == "./clips_output"
        assert cfg.save_analysis is True
        assert cfg.verbose is True

    def test_nested_configs(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.transcriber, TranscriberConfig)
        assert isinstance(cfg.analyzer, AnalyzerConfig)
        assert isinstance(cfg.clipper, ClipperConfig)

    def test_custom_nested(self):
        cfg = PipelineConfig(
            transcriber=TranscriberConfig(model_size="tiny"),
            analyzer=AnalyzerConfig(max_clips=5),
        )
        assert cfg.transcriber.model_size == "tiny"
        assert cfg.analyzer.max_clips == 5
