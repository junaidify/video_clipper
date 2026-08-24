"""
Audio Processing module: text-to-speech synthesis, commentary mixing, background music search and ducking.
"""
from video_clipper.audio_processing.tts_engine import (
    TTSResult,
    get_available_voices,
    synthesize_commentary,
    DEFAULT_VOICE,
    VOICE_PRESETS,
)
from video_clipper.audio_processing.audio_mixer import (
    AudioMixResult,
    mix_commentary,
)
from video_clipper.audio_processing.music_mixer import (
    MusicTrack,
    MusicMixResult,
    search_pixabay_music,
    get_local_music,
    download_music,
    mix_background_music,
    generate_silent_tone,
    get_mood_options,
)

__all__ = [
    "TTSResult",
    "get_available_voices",
    "synthesize_commentary",
    "DEFAULT_VOICE",
    "VOICE_PRESETS",
    "AudioMixResult",
    "mix_commentary",
    "MusicTrack",
    "MusicMixResult",
    "search_pixabay_music",
    "get_local_music",
    "download_music",
    "mix_background_music",
    "generate_silent_tone",
    "get_mood_options",
]
