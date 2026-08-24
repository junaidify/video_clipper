"""Unit tests for newly modularized video_clipper subpackages."""
import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_clipper.content_generation.script_generator import (
    Scene, Script, get_style_presets, _parse_script_json
)
from video_clipper.content_generation.trend_scout import (
    TrendingTopic, get_available_categories, _categorize_topic, _extract_keywords, _parse_traffic_score
)
from video_clipper.audio_processing.tts_engine import (
    TTSResult, get_available_voices, _enhance_text
)
from video_clipper.audio_processing.music_mixer import (
    MusicTrack, get_mood_options
)
from video_clipper.video_processing.video_modulator import (
    ModulationConfig, get_presets
)
from video_clipper.video_processing.video_editor import (
    EditConfig, EditResult, _map_transition
)
from video_clipper.media_management.library import (
    VideoEntry, VideoLibrary
)
from video_clipper.pattern_learning.trainer import (
    TrainingProfile, PatternTrainer
)


class TestContentGeneration:
    def test_script_and_scene_models(self):
        scene = Scene(
            scene_number=1, start_time=0.0, end_time=5.0,
            narration="Hook line", visual_description="Action",
            visual_keywords=["action", "fast"], text_overlay="HOOK"
        )
        assert scene.duration == 5.0
        assert scene.to_dict()["scene_number"] == 1

        script = Script(
            title="Viral Short", topic="AI", hook="Hook line",
            scenes=[scene], target_duration=45
        )
        assert script.total_duration == 5.0
        assert script.to_dict()["title"] == "Viral Short"

    def test_parse_script_json(self):
        raw_json = json.dumps({
            "title": "Test Title",
            "hook": "Test Hook",
            "scenes": [
                {
                    "scene_number": 1,
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "narration": "Line 1",
                    "visual_description": "Desc 1",
                    "visual_keywords": ["k1", "k2"],
                }
            ]
        })
        parsed = _parse_script_json(raw_json, "Topic", 45, "energetic", "informative")
        assert parsed is not None
        assert parsed.title == "Test Title"
        assert len(parsed.scenes) == 1

    def test_trend_scout_helpers(self):
        assert _categorize_topic("new AI model released") == "tech"
        assert _categorize_topic("bitcoin crypto bull market") == "finance"
        assert _categorize_topic("gym workout protein diet") == "health"

        kws = _extract_keywords("This is an extraordinary revolutionary breakthrough")
        assert "extraordinary" in kws
        assert "revolutionary" in kws

        score = _parse_traffic_score("500K+")
        assert score > 50.0

        cats = get_available_categories()
        assert len(cats) >= 5


class TestAudioProcessing:
    def test_voice_presets(self):
        voices = get_available_voices()
        assert len(voices) >= 10
        assert any(v["key"] == "andrew" for v in voices)

    def test_enhance_text_emphasis(self):
        enhanced = _enhance_text("This is an incredible and unstoppable victory.")
        assert "INCREDIBLE" in enhanced
        assert "UNSTOPPABLE" in enhanced
        assert "VICTORY" in enhanced

    def test_music_mood_options(self):
        moods = get_mood_options()
        assert len(moods) >= 5
        assert any(m["id"] == "upbeat" for m in moods)


class TestVideoProcessing:
    def test_modulation_presets(self):
        presets = get_presets()
        assert "stealth" in presets
        assert "cinematic" in presets
        assert "vibrant" in presets
        assert "warm_flip" in presets
        assert "maximum" in presets

    def test_edit_config_defaults(self):
        cfg = EditConfig()
        assert cfg.width == 1080
        assert cfg.height == 1920
        assert cfg.fps == 30
        assert cfg.color_correction is True
        assert cfg.ken_burns is True

    def test_map_transition(self):
        assert _map_transition("fade") == "fade"
        assert _map_transition("slide_left") == "slideleft"
        assert _map_transition("zoom") == "smoothup"


class TestMediaManagement:
    def test_library_operations(self, tmp_path):
        lib_dir = tmp_path / "test_lib"
        lib = VideoLibrary(str(lib_dir))

        # Create dummy video file
        dummy_vid = tmp_path / "dummy.mp4"
        dummy_vid.write_bytes(b"dummy data 12345678")

        entry = lib.add_video(str(dummy_vid), "Sample Video", source="upload")
        assert entry.title == "Sample Video"
        assert os.path.isfile(entry.file_path)

        retrieved = lib.get_video(entry.video_id)
        assert retrieved is not None
        assert retrieved.title == "Sample Video"

        stats = lib.get_library_stats()
        assert stats["total_videos"] == 1

        lib.delete_video(entry.video_id)
        assert lib.get_video(entry.video_id) is None


class TestPatternLearning:
    def test_trainer_session_and_profile(self, tmp_path):
        sessions_dir = tmp_path / "training"
        trainer = PatternTrainer(str(sessions_dir))

        session_id = trainer.create_session()
        assert session_id is not None

        trainer.add_short_form(session_id, {
            "duration": 45.0,
            "full_text": "Here is the number one secret to winning every day.",
            "segments": [{"text": "Here is the number one secret", "start": 0.0, "end": 4.0}],
        })

        profile = trainer.extract_patterns(session_id)
        assert profile is not None
        assert profile.short_form_count == 1
        assert "secret" in profile.opening_keywords or "winning" in profile.opening_keywords

        saved_profile = trainer.get_profile(session_id)
        assert saved_profile is not None
        assert saved_profile.profile_id == session_id

        assert trainer.delete_session(session_id) is True
