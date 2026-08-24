"""
Pattern Trainer Module
Upload long-form + short-form video examples to extract creator-specific clipping patterns.
Learned patterns are serialized as a private JSON profile and incorporated into ContentAnalyzer scoring.
"""
import json
import logging
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


@dataclass
class TrainingProfile:
    """Extracted patterns and preferences from training examples."""
    profile_id: str
    created_at: str
    avg_clip_duration: float = 0.0
    min_clip_duration: float = 0.0
    max_clip_duration: float = 0.0
    position_distribution: dict = field(default_factory=dict)
    opening_keywords: List[str] = field(default_factory=list)
    content_keywords: List[str] = field(default_factory=list)
    avg_words_per_second: float = 0.0
    long_form_count: int = 0
    short_form_count: int = 0
    clip_durations: List[float] = field(default_factory=list)
    clip_positions: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Training profile saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TrainingProfile":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


STOP_WORDS = set("""
a an the is are was were be been being have has had do does did will would
shall should may might can could must i me my we us our you your he him his
she her it its they them their what which who whom this that these those
am is are was were been being have has had having do does did doing and but
if or because as until while of at by for with about between through during
before after above below to from up down in out on off over under again
further then once here there when where why how all both each few more most
other some such no nor not only own same so than too very just don should
now like yeah yes okay ok well so um uh really actually know think mean
right going gonna wanna got get let say said thing things go went come
""".split())


class PatternTrainer:
    """Extracts clipping patterns from example videos."""

    def __init__(self, sessions_dir: str):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())[:8]
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "long_form_transcripts": [],
            "short_form_transcripts": [],
            "profile": None,
        }
        with open(session_dir / "session.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

        logger.info(f"Training session created: {session_id}")
        return session_id

    def add_long_form(self, session_id: str, transcript_data: dict):
        session_dir = self.sessions_dir / session_id
        session_file = session_dir / "session.json"

        if not session_file.exists():
            raise ValueError(f"Session not found: {session_id}")

        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["long_form_transcripts"].append({
            "added_at": datetime.now().isoformat(),
            "duration": transcript_data.get("duration", 0),
            "segments": transcript_data.get("segments", []),
            "full_text": transcript_data.get("full_text", ""),
        })

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Long-form added to session {session_id}")

    def add_short_form(self, session_id: str, transcript_data: dict):
        session_dir = self.sessions_dir / session_id
        session_file = session_dir / "session.json"

        if not session_file.exists():
            raise ValueError(f"Session not found: {session_id}")

        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["short_form_transcripts"].append({
            "added_at": datetime.now().isoformat(),
            "duration": transcript_data.get("duration", 0),
            "segments": transcript_data.get("segments", []),
            "full_text": transcript_data.get("full_text", ""),
        })

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Short-form added to session {session_id}")

    def extract_patterns(self, session_id: str) -> TrainingProfile:
        session_dir = self.sessions_dir / session_id
        session_file = session_dir / "session.json"

        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        long_forms = data.get("long_form_transcripts", [])
        short_forms = data.get("short_form_transcripts", [])

        profile = TrainingProfile(
            profile_id=session_id,
            created_at=datetime.now().isoformat(),
            long_form_count=len(long_forms),
            short_form_count=len(short_forms),
        )

        if short_forms:
            durations = [sf["duration"] for sf in short_forms if sf.get("duration")]
            if durations:
                profile.clip_durations = durations
                profile.avg_clip_duration = sum(durations) / len(durations)
                profile.min_clip_duration = min(durations)
                profile.max_clip_duration = max(durations)

            opening_words = Counter()
            all_words = Counter()
            for sf in short_forms:
                segments = sf.get("segments", [])
                text = sf.get("full_text", "")

                if segments:
                    first_text = segments[0].get("text", "")
                    tokens = self._tokenize(first_text)
                    opening_words.update(tokens)

                tokens = self._tokenize(text)
                all_words.update(tokens)

                if sf.get("duration") and text:
                    word_count = len(text.split())
                    wps = word_count / sf["duration"]
                    profile.avg_words_per_second = (
                        (profile.avg_words_per_second + wps) / 2
                        if profile.avg_words_per_second > 0 else wps
                    )

            profile.opening_keywords = [word for word, _ in opening_words.most_common(30)]
            profile.content_keywords = [word for word, _ in all_words.most_common(50)]

        if long_forms and short_forms:
            profile.position_distribution = self._analyze_positions(long_forms, short_forms)
            profile.clip_positions = list(profile.position_distribution.get("positions", []))

        profile.save(str(session_dir / "profile.json"))

        data["profile"] = profile.to_dict()
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Patterns extracted for session {session_id}")
        return profile

    def get_profile(self, session_id: str) -> Optional[TrainingProfile]:
        profile_path = self.sessions_dir / session_id / "profile.json"
        if profile_path.exists():
            return TrainingProfile.load(str(profile_path))
        return None

    def list_sessions(self) -> List[dict]:
        sessions = []
        for d in self.sessions_dir.iterdir():
            if d.is_dir():
                session_file = d / "session.json"
                if session_file.exists():
                    try:
                        with open(session_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        sessions.append({
                            "session_id": data["session_id"],
                            "created_at": data["created_at"],
                            "long_form_count": len(data.get("long_form_transcripts", [])),
                            "short_form_count": len(data.get("short_form_transcripts", [])),
                            "has_profile": data.get("profile") is not None,
                        })
                    except Exception:
                        pass
        sessions.sort(key=lambda s: s["created_at"], reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        import shutil
        session_dir = self.sessions_dir / session_id
        if session_dir.exists():
            shutil.rmtree(str(session_dir), ignore_errors=True)
            return True
        return False

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'[a-z]+', text.lower())
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    def _analyze_positions(self, long_forms: list, short_forms: list) -> dict:
        positions = []
        buckets = {"0-20%": 0, "20-40%": 0, "40-60%": 0, "60-80%": 0, "80-100%": 0}

        for lf in long_forms:
            lf_text = lf.get("full_text", "").lower()
            lf_duration = lf.get("duration", 1)
            lf_segments = lf.get("segments", [])

            if not lf_text or not lf_segments:
                continue

            for sf in short_forms:
                sf_text = sf.get("full_text", "").lower()
                if not sf_text:
                    continue

                search_key = sf_text[:80]
                idx = lf_text.find(search_key)

                if idx >= 0:
                    char_ratio = idx / max(len(lf_text), 1)
                    time_pos = char_ratio * lf_duration
                    pct = char_ratio * 100
                    positions.append(round(pct, 1))

                    if pct < 20:
                        buckets["0-20%"] += 1
                    elif pct < 40:
                        buckets["20-40%"] += 1
                    elif pct < 60:
                        buckets["40-60%"] += 1
                    elif pct < 80:
                        buckets["60-80%"] += 1
                    else:
                        buckets["80-100%"] += 1

        return {
            "positions": positions,
            "distribution": buckets,
        }
