"""
Video Library Module
Persistent storage, metadata index, and asset repository for uploaded/downloaded videos.
Each video gets its own directory: {sanitized_title}_{YYYYMMDD_HHMMSS}
"""
import json
import logging
import os
import re
import shutil
import threading
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

LIBRARY_INDEX = "library_index.json"


@dataclass
class VideoEntry:
    """A single video asset in the library."""
    video_id: str
    title: str
    file_path: str
    directory: str
    source: str
    source_url: Optional[str] = None
    duration: Optional[float] = None
    added_at: str = ""
    file_size_mb: float = 0.0
    uploader: Optional[str] = None
    channel_url: Optional[str] = None
    clips_directories: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class VideoLibrary:
    """Manages persistent video storage and retrieval."""

    def __init__(self, library_root: str):
        self.root = Path(library_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / LIBRARY_INDEX
        self._lock = threading.Lock()
        self._index = self._load_index()

    def _load_index(self) -> dict:
        if self._index_path.exists():
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Corrupted library index, starting fresh")
        return {"videos": {}}

    def _save_index(self):
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _sanitize_name(title: str) -> str:
        safe = re.sub(r'[^\w\s-]', '', title)
        safe = re.sub(r'\s+', '_', safe.strip())
        safe = safe[:60] if len(safe) > 60 else safe
        return safe or "video"

    @staticmethod
    def _generate_video_id(title: str) -> str:
        safe_title = VideoLibrary._sanitize_name(title)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_title}_{timestamp}"

    def add_video(
        self,
        source_path: str,
        title: str,
        source: str = "upload",
        source_url: Optional[str] = None,
        duration: Optional[float] = None,
        uploader: Optional[str] = None,
        channel_url: Optional[str] = None,
    ) -> VideoEntry:
        """Add a video to the library by moving/copying into its isolated directory."""
        video_id = self._generate_video_id(title)
        video_dir = self.root / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        src = Path(source_path)
        dest = video_dir / src.name
        shutil.copy2(str(src), str(dest))

        file_size_mb = dest.stat().st_size / (1024 * 1024)

        entry = VideoEntry(
            video_id=video_id,
            title=title,
            file_path=str(dest),
            directory=str(video_dir),
            source=source,
            source_url=source_url,
            duration=duration,
            added_at=datetime.now().isoformat(),
            file_size_mb=round(file_size_mb, 2),
            uploader=uploader or None,
            channel_url=channel_url or None,
            clips_directories=[],
        )

        with self._lock:
            self._index["videos"][video_id] = entry.to_dict()
            self._save_index()

        logger.info(f"Video added to library: {video_id} ({file_size_mb:.1f} MB)")
        return entry

    def get_video(self, video_id: str) -> Optional[VideoEntry]:
        data = self._index["videos"].get(video_id)
        if not data:
            return None
        return VideoEntry(**data)

    def list_videos(self) -> List[VideoEntry]:
        entries = []
        for vid_data in self._index["videos"].values():
            entry = VideoEntry(**vid_data)
            if Path(entry.file_path).exists():
                entries.append(entry)

        entries.sort(key=lambda e: e.added_at, reverse=True)
        return entries

    def add_clips_directory(self, video_id: str, clips_dir: str):
        with self._lock:
            if video_id in self._index["videos"]:
                if clips_dir not in self._index["videos"][video_id]["clips_directories"]:
                    self._index["videos"][video_id]["clips_directories"].append(clips_dir)
                    self._save_index()

    def get_clips_for_video(self, video_id: str) -> List[str]:
        data = self._index["videos"].get(video_id, {})
        dirs = data.get("clips_directories", [])
        return [d for d in dirs if Path(d).exists()]

    def delete_video(self, video_id: str) -> bool:
        with self._lock:
            data = self._index["videos"].get(video_id)
            if not data:
                return False

            video_dir = Path(data["directory"])
            if video_dir.exists():
                shutil.rmtree(str(video_dir), ignore_errors=True)

            del self._index["videos"][video_id]
            self._save_index()
            logger.info(f"Video removed from library: {video_id}")
            return True

    def get_library_stats(self) -> dict:
        videos = self.list_videos()
        total_size = sum(v.file_size_mb for v in videos)
        total_clips = sum(len(v.clips_directories) for v in videos)
        return {
            "total_videos": len(videos),
            "total_size_mb": round(total_size, 2),
            "total_clip_sessions": total_clips,
        }
